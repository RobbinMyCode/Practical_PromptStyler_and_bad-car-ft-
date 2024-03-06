import os
import argparse
import torch
import clip
from torch import nn
from torch.nn import functional as F
from datetime import datetime
from image_loader import CheapTestImageDataset
import pseudoCLIP
import numpy as np
import pickle

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch CLIP distillation")
    parser.add_argument("--dataset", default="PACS")
    parser.add_argument("--Domain_ID", default=['sketch', 'photo', 'cartoon', 'art_painting'])
    parser.add_argument("--classes", default=["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscale")
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs for each styleword")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", default="resnetv2_50x1_bit.goog_in21k_ft_in1k", help="Which network to use")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='', help="Used by the logger to save logs")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--CLIP", default="ViT-B/16", help="CLIP model")
    parser.add_argument("--output_folder", default='run1', help="folder where to save results file")
    parser.add_argument("--output_file_name", default='.txt', help="results file name")
    parser.add_argument("--data_path", default='', help="path of the dataset")
    parser.add_argument("--number_style_words", "-n", default=1, help="number of stylewords for ech pseudo length to train")
    parser.add_argument("--pseudo_lengths", default=[1], help="length of sequences pseudowords are trained for")
    parser.add_argument("--class_words_index", "-ci", default=[[1]], help="indizes where to put class, poss values = 'all': every position, or a list of lists (for each pseudo_length) e.g [[0,1,2], [2]]")
    #parser.add_argument("--style_word_basis", default='a photo of a', help="wordbasis for which stylewords are created, photo --> pseudoword")
    #parser.add_argument("--style_word_index", default=1,
    #                    help="index of which word in style_word_basis shall be replaced by pseudoword; must be int in [0, len(style_word_basis)]")
    parser.add_argument("--save_style_words", default="yes", help='''if 'yes' saves the style-context words as /saved_prompts/[dataset]_[class]_[CLIP model].pickle,
                                                    if 'extend' extends the style-context words in /saved_prompts/[dataset]_[class]_[CLIP model].pickle by the newly created ones.
                                                    saves are as numpy arrays''')



    return parser.parse_args()


class WordModel(nn.Module):
    def __init__(self, pseudoCLIP_model, class_word_index=[[0,1]], pseudo_lengths=[1], n_style_words_per_config=5, style_word_dim=512, n_classes=0, device="cuda"):
        super(WordModel, self).__init__()

        length = np.max(pseudo_lengths)
        self.style_words = torch.nn.Parameter((torch.randn((length, len(pseudo_lengths), length, n_style_words_per_config, style_word_dim), dtype=torch.float16)) * 0.001)#0.02 #adding promtstyler
        #has potentially more entries in dim=2 than nessesairy --> MUST BE CONSIDERED for later steps (aka these entries ignored)
        self.style_words.requires_grad = True

        #for encoding
        self.pseudo_clip_encoder = pseudoCLIP_model
        for name, param in self.pseudo_clip_encoder.named_parameters():
            param.requires_grad = False

        self.cw_index = class_word_index
        self.pseudo_length = pseudo_lengths
        self.device = device
        self.n_style = n_style_words_per_config
        self.encode_dim = style_word_dim
        self.n_classes = n_classes
        with torch.no_grad():
            self.pseudo_clip_encoder.eval()

            #self.style_dummy_token = clip.tokenize(self.basic_phrase).to(self.device).detach()
    def forward(self, content_words, style_index=1):
        if self.n_classes == 0:
            self.n_classes = len(content_words)

        #-- style_featuers = style_words flattened across first 3 dimensions
        #print("cw index", self.cw_index)
        #====================================
        #storage efficient saving of style / style_content vectors, no need for seperation like this
        style_content_features = torch.randn(
            (np.max(self.pseudo_length), len(self.pseudo_length), np.max(self.pseudo_length), self.n_style, self.n_classes, self.encode_dim),
            dtype=torch.float16).to(self.device)
        style_features = torch.randn(
            (np.max(self.pseudo_length), len(self.pseudo_length), np.max(self.pseudo_length), self.n_style, self.encode_dim),
            dtype=torch.float16).to(self.device)

        for pseudo_idx, pseudo_length in enumerate(self.pseudo_length):
            for c_count, c_idx in enumerate(self.cw_index[pseudo_idx]):
                for n_cont, content_word in enumerate(content_words):
                    # -- STYLE CONTENT WORDS
                    text = "a "* c_idx + content_word + " a" * max(pseudo_length-c_idx, 0) #-- "a" is an example word, doesnt matter which,could also be "cat"
                    sc_token = clip.tokenize(text).to(self.device)

                    positions_pseudo = [i for i in range(pseudo_length)]
                    if c_idx < pseudo_length:
                        positions_pseudo[max(c_idx-1,0):] = list(np.array(positions_pseudo[max(c_idx-1,0):]) + 1) #positions_pseudo + [pseudo_length + 1]

                    for n_vector in range(self.n_style):
                        #print(c_idx, pseudo_idx, pseudo_length, n_vector)
                        #print(self.style_words.size())
                        sc_feat = self.pseudo_clip_encoder.encode_text_multiple_pseudo(
                                                    sc_token,
                                                    pseudowords=self.style_words[c_idx, pseudo_idx, :pseudo_length, n_vector],
                                                    positions_pseudo=positions_pseudo)[None, None, :, None, None, :]
                        style_content_features[c_count, pseudo_idx, :np.max(self.pseudo_length), n_vector, n_cont, :] = sc_feat
                        style_content_features[c_count, pseudo_idx, np.max(self.pseudo_length):, n_vector, n_cont, :] = 0 #all non needed = 0, --> no contribiution to style loss

                    # -- STYLE WORDS
                    if n_cont == 0: #ugly way to out out of loop
                        style_text = "a" * pseudo_length
                        s_token = clip.tokenize(style_text).to(self.device)

                        for n_vector in range(self.n_style):
                            s_feat = self.pseudo_clip_encoder.encode_text_multiple_pseudo(
                                                    s_token,
                                                    pseudowords=self.style_words[c_idx, pseudo_idx, :pseudo_length, n_vector],
                                                    positions_pseudo=positions_pseudo)[None, None, :, None, :]
                            style_features[c_count, pseudo_idx, :np.max(self.pseudo_length), n_vector, :] = s_feat
                            style_features[c_count, pseudo_idx, np.max(self.pseudo_length):, n_vector, :] = 0

        if torch.sum(torch.isnan(style_content_features)) >=1:
            print("sc token", sc_token)
            print("used style words", self.style_words[:style_index+1])
            print("style-content features", style_content_features)
            print("w.o. to float32", self.pseudo_clip_encoder.encode_text(sc_token, self.style_words[:style_index+1], position_pseudo=self.index, print_intermediate=True))
            torch.save(self.style_words[:style_index+1], "values.pt")
            exit()
        #style_features = torch.flatten(self.style_words, end_dim=2)
        return [style_features, style_content_features]

class ArcFaceLinear(nn.Module):
    def __init__(self, dim_in, dim_out, s=5, m=0.5):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out, bias=False)
    def forward(self, style_content_words):
        x = self.fc(style_content_words)
        return x, self.fc.weight.detach().clone()

def style_loss(style_vec, style_index=1):
    if style_index == 0:
        return 0
    else:
        loss = 0

        #norm_i = torch.linalg.norm(style_vec[style_index])
        #for j in range(0, style_index): orthogonal TO ALL
        norms = [torch.linalg.norm(s_v_j) for s_v_j in style_vec]
        for j in range(0, len(style_vec)):
            for i in range(0, j):
                loss += torch.abs(1./j * style_vec[j] @ style_vec[i] /
                         (norms[j] * norms[i]))
        loss /= len(style_vec)
        return loss

def content_loss(style_content_words, content_words, style_index=1, n_classes=7):
    '''
        loss to maximize cosine similarity with style_content_vector and corresponding content_vector while minimizing all other similarities
        - exp for softmax and log for loss scaling [0,1]-> [infty, 0]
    '''
    ##-- no use of style_index
    loss = 0
    for idx in range(len(style_content_words)):
        z = style_content_words[idx] @ content_words.T / (
                torch.linalg.norm(style_content_words[idx], axis=-1) * torch.linalg.norm(content_words, axis=-1))
        #z = style_content_words[style_index] @ content_words.T / (
         #       torch.linalg.norm(style_content_words[style_index], axis=-1) * torch.linalg.norm(content_words, axis=-1))
        z = torch.exp(z)
        sum_z_imn = torch.sum(z, dim=-1)

        z_diag = torch.diagonal(z, 0)

        loss += -1. / n_classes* torch.sum( torch.log(z_diag) - torch.log(sum_z_imn) )
    loss /= len(style_content_words)
    return loss

def style_content_loss(model_output, content_labels, style_index=1, n_classes=7):
    loss = content_loss(model_output[1], content_labels.to(torch.float16), style_index, n_classes) + 0.1*style_loss(model_output[0], style_index)
    #loss += style_loss(model_output[2], style_index)
    if torch.isnan(loss):
        print(loss)
        exit("nan loss occured")
    return loss

def arcface_loss(linear_output, linear_input, linear_weights, labels, s=5, m=0.5):
    #arcface_softmax
    cos_m = torch.cos(torch.tensor(m))
    sin_m = torch.sin(torch.tensor(m))

    norm_weights = torch.sqrt(torch.sum(linear_weights ** 2, axis=-1)).unsqueeze(0)
    norm_input = torch.sqrt(torch.sum(linear_input ** 2, axis=-1)).unsqueeze(-1)

    # from out = in @ weights = |in| * |weights| *cos(theta) --> cos(theta) = out/(|in| * |weights|)
    cos_theta = linear_output / (norm_input * norm_weights)

    sin_theta = (1 - cos_theta ** 2) ** 0.5
    cos_theta_plus_m = cos_theta * cos_m - sin_theta * sin_m  # expansion formula of cos(a+b)

    exp_pos = torch.sum(torch.exp(s * cos_theta_plus_m) * F.one_hot(labels, num_classes=-1), axis=-1)
    exp_neg = torch.exp(s * cos_theta)

    #denominator_sum_all = torch.sum(exp_neg, axis=-1)
    denominator_sum = torch.sum(exp_neg * (1 - F.one_hot(labels, num_classes=-1)), axis=-1)

    target = exp_pos / (denominator_sum + exp_pos)
    error = - torch.log(target)
    return torch.mean(error)
def arcface_softmax(linear_output, linear_input, linear_weights, s=5, m=0.5):
    '''
        computes the arcface softmax of a linear layer (which must not use a bias)
    :param self:
    :param linear_output: output of linear layer (of which to compute the arcface softmax)
    :param linear_input: input to linear layer
    :param linear_weights: linear.weight (for linear layer)
    :param s: scaling hyperparameter of arcface
    :param m: angle parameter of arcface
    :return:
    '''
    cos_m = torch.cos(torch.tensor(m))
    sin_m = torch.sin(torch.tensor(m))

    norm_weights = torch.sqrt(torch.sum(linear_weights ** 2, axis=-1)).unsqueeze(0)
    norm_input = torch.sqrt(torch.sum(linear_input ** 2, axis=-1)).unsqueeze(-1)

    #from out = in @ weights = |in| * |weights| *cos(theta) --> cos(theta) = out/(|in| * |weights|)
    cos_theta = linear_output / (norm_input * norm_weights)

    sin_theta = (1 - cos_theta ** 2) ** 0.5
    cos_theta_plus_m = cos_theta * cos_m - sin_theta * sin_m  #expansion formula of cos(a+b)

    exp_pos = torch.exp(s * cos_theta_plus_m)
    exp_neg = torch.exp(s * cos_theta)

    denominator_sum_all = torch.sum(exp_neg, axis=-1)
    result = exp_pos

    for i in range(result.size(1)):
        result[:, i] /= (denominator_sum_all - exp_neg[:, i] + exp_pos[:, i]) # / all neg but i
    #for i in range(15):  # resultb.size(0)):
    #print(result[:5])
    return result

class Trainer:
    def __init__(self, args, device, target_name):
        self.args = args
        self.device = device

        self.clip_model, self.image_preprocess = pseudoCLIP.load(self.args.CLIP, device=self.device)
        if self.args.dataset == "Terra":
            print("please load your finetuned CLIP weight here")
            # model_weights = torch.load("/path/finetuned_clip")
            # self.clip_model.load_state_dict(model_weights)
        self.text_feature_dim = 512  if args.CLIP == "ViT-B/16" else 768#=1024 resnet, =512 Vit-B/16, #768 viT-L/14
        self.n_style_words = self.args.number_style_words


        # -- get class_words_index in a useful format
        if isinstance(args.class_words_index, str):
            #if args.class_words_index == "all": -- all strings will do 'all' at this point
            args.class_words_index = [[i for i in range(k)] for k in args.pseudo_lengths]
        elif isinstance(args.class_words_index, list): #and all(isinstance(x, int) for x in args.class_words_index):
            for pseudo_idx, pseudo_length in enumerate(args.pseudo_lengths):
                for i in range(len(args.class_words_index)): #only positive indize so every index is unique for a position
                    if args.class_words_index[pseudo_idx][i] < 0:
                        args.class_words_index[pseudo_idx][i] += args.pseudo_length
                        args.class_words_index[pseudo_idx] = list(set(args.class_words_index[pseudo_idx])) #remove dublicates
                if len(args.class_words_index) > pseudo_length + 1:
                    print("Warning: class word index out of range (max=", args.pseudo_length + 1,") will be ignored: ", sep="", end="")
                    for cw_idx in args.class_words_index:
                        print(cw_idx, end=", ")
                        args.class_words_index.remove(cw_idx)
        else:
            exit("class words index is neither 'all' nor a list of integer")


        if args.dataset == "Terra":
            self.text_anchor = ['bright photo', 'corrupted photo', 'dark photo', 'good photo']
        elif args.dataset == "VLCS":
            self.text_anchor = ['bright photo', 'corrupted photo', 'dark photo', 'good photo']
        else:
            self.text_anchor = args.source  ###all named styles except the one for testing e.g. could be PACS: ['art_painting', 'sketch', 'photo'] but not 'cartoon'

        token_classes = torch.cat([clip.tokenize(f"{c}") for c in self.args.classes]).to(self.device)
        self.content_features = self.clip_model.encode_text(token_classes).to(torch.float32).to(self.device)
        with torch.no_grad():
            self.clip_model.eval()


        word_model = WordModel(self.clip_model, class_word_index=args.class_words_index, pseudo_lengths=args.pseudo_lengths,
                               n_style_words_per_config=args.number_style_words, style_word_dim=self.text_feature_dim,
                               n_classes=len(args.classes), device="cuda")
        self.word_model = word_model.to(self.device)


        self.test_data = CheapTestImageDataset(base_path="/home/robin/Documents/Domain_Generalization/data/"+args.dataset,
                                    domains=args.target, class_names=self.args.classes)
        self.dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=args.batch_size, shuffle=True)
        self.optimizer_sgd = torch.optim.SGD(word_model.parameters(),  momentum=0.9, lr=0.002)
        self.lin_epochs = 100

        self.current_epoch = 0
        self.target_name = target_name
    def _do_epoch(self):
        '''
            trains an epoch for pseudowords
        :return:
        '''

        self.word_model.train()
        self.word_model.zero_grad()

        class_words = self.args.classes #torch.cat((self.args.classes, self.content_features_style_word), 0)[torch.randint(len(self.args.classes), (128,))]
        self.optimizer_sgd.zero_grad()
        model_output = self.word_model(class_words, style_index=self.n_style_vec)
        #print("og vs flat: [0]", model_output[0].size(), torch.flatten(model_output[0], end_dim=3).size())
        #print("og vs flat: [1]", model_output[1].size(), torch.flatten(model_output[1], end_dim=3).size())
        model_output[0] = torch.flatten(model_output[0], end_dim=3)
        model_output[1] = torch.flatten(model_output[1], end_dim=3) #e.g. 2,2,2,5,10,512 -> 40, 10, 512
        #print("og vs flat", model_output[1].size(), torch.flatten(model_output[1], end_dim=-2).size())
        word_loss = style_content_loss(model_output, self.content_features,
                                       style_index=self.n_style_vec, n_classes=len(self.args.classes))
        if self.current_epoch == self.args.epochs and self.n_style_vec == self.n_style_words -1:
            word_loss.backward(retain_graph=False)
        else:
            word_loss.backward(retain_graph=True)
        self.optimizer_sgd.step()
        self.word_model.eval()
        print('|', end='', sep='')
        return model_output

    def do_test(self, features, labels, paths):
        from PIL import ImageFile, Image
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        import numpy as np
        class_correct = 0
        data, class_l = features.to(self.device), labels.to(self.device)

        for i, path in enumerate(paths):
            data_n = self.image_preprocess(Image.open(path)).to(self.device).unsqueeze(0)
            if i == 0:
                CLIP_image_features = self.clip_model.encode_image(data_n).type(torch.float32).to(self.device)
            else:
                CLIP_image_features = torch.cat((CLIP_image_features, self.clip_model.encode_image(data_n).type(torch.float32).to(self.device)), 0)

        predictions, _ = self.lin_model(CLIP_image_features)
        softmax_pred = torch.nn.Softmax(dim=-1)(predictions)

        pred_index = torch.argmax(softmax_pred, axis=-1)
        class_correct += torch.sum(pred_index == class_l)
        return class_correct

    def do_training(self):
        #token_style_word_to_class = torch.cat([clip.tokenize(f"{self.args.style_word_basis} {c}.") for c in self.args.classes]).to(
        #   self.device)
        #self.content_features_style_word = self.clip_model.encode_text(token_style_word_to_class).to(torch.float32).to(
        #   self.device)


        train_noise = True
        #print("init parameter", self.word_model.style_words)
        if train_noise:
            for self.n_style_vec in range(self.n_style_words):
                for self.current_epoch in range(self.args.epochs):
                    m_o = self._do_epoch()

                print("training of style_word #" + str(self.n_style_vec) + " finished")


        _ , final_style_content_words = self.word_model(self.args.classes)
        flat_style_content = torch.flatten(final_style_content_words, end_dim=3)

        if self.args.save_style_words == "yes" or self.args.save_style_words == "extend":
            with torch.no_grad():
                for idx, class_X in enumerate(self.args.classes):
                    full_vecs = flat_style_content[:, idx, :]

                    pref = 'saved_prompts/'
                    path_p2 =  self.args.dataset + "_" + class_X + "_" + self.args.CLIP + ".pickle"
                    path = pref + path_p2.replace("/","")

                    if self.args.save_style_words == "extend":
                        with open(path, 'rb') as handle: #with statement --> auto close
                            old_vecs = pickle.load(handle)
                        full_vecs = np.concatenate((old_vecs, full_vecs), axis=0)

                    with open(path,  'wb') as handle:
                        pickle.dump(full_vecs, handle, protocol=pickle.HIGHEST_PROTOCOL)

                print("Word vectors saved \n", "=======================================", sep="")





        #print(final_style_words)
        #exit()
        #description_list = [f"itap of a {c}.") for c in self.args.classes]

        RAM_Device = self.device if (self.args.dataset == "PACS" or self.args.dataset == "VLCS") else "cpu"


        input = torch.flatten(final_style_content_words, end_dim=4)
        targets = torch.tensor(range(len(self.content_features))).repeat( #1*len(flattened_sc_no)+
            len(input)//len(self.content_features), 1, 1).flatten().to(RAM_Device) # permute(1,0)


        self.input = input
        self.lin_model = ArcFaceLinear(self.text_feature_dim, len(self.content_features)).to(self.device)
        lin_optimizer = torch.optim.SGD(self.lin_model.parameters(), lr=0.005, momentum=0.9)


        #lin layer training
        batch_size_to_remember = 128 if len(input)>128 else len(input)#32 if self.args.dataset == "Officehome" else 128
        batchsize = batch_size_to_remember
        for n_lin_epoch in range(self.lin_epochs):#15):#2*2*self.lin_epochs):
            for batch_start in range(0, input.size(0), batchsize):
                self.lin_model.train()
                self.lin_model.zero_grad()
                lin_optimizer.zero_grad()
                #self.optimizer_ArcFace.zero_grad()
                if batch_start+batchsize > input.size(0):   # end at end of input not beyond
                    batchsize = input.size(0) - batch_start

                #torch.autograd.set_detect_anomaly(True)
                randperm = torch.randperm(batchsize)
                batch_in = (input[batch_start : batch_start+batchsize])[randperm].to(self.device).to(torch.float32)
                batch_target = (targets[batch_start : batch_start+batchsize])[randperm].to(self.device)
                lin_model_pred, weights = self.lin_model(batch_in)

                #softmax_pred = arcface_softmax(lin_model_pred, batch_in, weights)
                loss_af = arcface_loss(lin_model_pred, batch_in, weights, batch_target)
                loss = 0 * torch.nn.CrossEntropyLoss()(lin_model_pred, batch_target) + 1 * loss_af#self.lin_model.arcface_loss(lin_model_pred, batch_target)
                print(loss.cpu().detach().numpy(), torch.nn.CrossEntropyLoss()(lin_model_pred, batch_target).cpu().detach().numpy(), sep="; ", end="||")
                loss.backward(retain_graph=True)
                lin_optimizer.step()
                batchsize = batch_size_to_remember
                #batch_in.detach()
                #batch_target.detach()
                if RAM_Device == "cpu":
                    for itm in [lin_model_pred, weights, batch_in, batch_target]:
                        itm.detach()

                #torch.cuda.empty_cache()
            print("epoch "+ str(n_lin_epoch)+ " done.")
        self.lin_model.eval()

        class_correct = 0
        intermediate_total = 0
        self.clip_model.to(self.device)
        with torch.no_grad():
            for execution_nr in range(self.dataloader.__len__()):  # complex way to write sample ove rall samples
                features, labels, _,  paths = next(iter(self.dataloader))
                total = self.test_data.__len__()
                intermediate_total += len(labels)
                class_correct += self.do_test(features, labels, paths)
                print("currently ", class_correct.cpu().detach().numpy(), "ouf of ", intermediate_total, " correct. (", class_correct.cpu().detach().numpy()/(intermediate_total)*100,"%)" )
            print("total accuracy:", 1.0 * class_correct / total)
        # return


def train_with_sweep():
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_num
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:"+args.GPU_num if torch.cuda.is_available() else "cpu")
    select_txt = os.path.join(os.getcwd(), 'data', 'hp_search', args.dataset + '.txt')
    #print("parameter search space: ")
    #with open(select_txt, 'r') as ff:
    #    lines = ff.readlines()
    #    print(lines)


    if args.dataset == "PACS":
        args.Domain_ID = ['art_painting', 'sketch', 'photo', 'cartoon']
        args.classes = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        args.n_classes = 7
        args.n_domain = 4
    elif args.dataset == "VLCS":
        args.Domain_ID = ["LabelMe", "SUN09", "VOC2007", "Caltech101"]
        args.classes = ["bird", "car", "chair", "dog", "person"]
        args.n_classes = 5
        args.n_domain = 4
    elif args.dataset == "Terra":
        args.Domain_ID = ["location_100", "location_38", "location_43", "location_46"]
        args.classes = ["bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit", "raccoon", "squirrel"]
        args.n_classes = 10
        args.n_domain = 4
        args.learning_rate = 0.002
    elif args.dataset == "Officehome":
        args.Domain_ID = ['Clipart', 'Art', 'RealWorld', 'Product']
        args.classes = ["Alarm_Clock", "Backpack", "Batteries", "Bed", "Bike", "Bottle", "Bucket", "Calculator",
                        "Calendar", "Candles", "Chair", "Clipboards", "Computer", "Couch", "Curtains", "Desk_Lamp",
                        "Drill", "Eraser", "Exit_Sign", "Fan", "File_Cabinet", "Flipflops", "Flowers", "Folder", "Fork",
                        "Glasses", "Hammer", "Helmet", "Kettle", "Keyboard", "Knives", "Lamp_Shade", "Laptop", "Marker",
                        "Monitor", "Mop", "Mouse", "Mug", "Notebook", "Oven", "Pan", "Paper_Clip", "Pen", "Pencil",
                        "Postit_Notes", "Printer", "Push_Pin", "Radio", "Refrigerator", "Ruler", "Scissors",
                        "Screwdriver", "Shelf", "Sink", "Sneakers", "Soda", "Speaker", "Spoon", "Table", "Telephone",
                        "ToothBrush", "Toys", "Trash_Can", "TV", "Webcam"]
        args.n_classes = 65
        args.n_domain = 4
    else:
        raise NotImplementedError

    #for domain in args.Domain_ID:
    args.target = args.Domain_ID #needs to stay for compatibility reasons with the rest of the rise code basis
    args.source = args.Domain_ID.copy()
    #args.source.remove(args.target) training does not use images, jus ttext, so no need to remove
    print("Training {} on source domains (text representation):".format(args.dataset))
    print(*args.source, sep=",")
    print("Test on target domains (images):")
    print(args.target)
    print("Classes are:", args.classes)

    now = datetime.now().strftime("%m-%d-%y_%H:%M:%S")
    output_file_name = now + '_' + args.dataset + ".txt"#'_' + args.target + '.txt'
    output_folder = os.path.join(os.getcwd(), 'results', args.output_folder)
    if os.path.exists(output_folder):
        pass
    else:
        os.makedirs(output_folder)
    args.output_file_name = os.path.join(output_folder, output_file_name)
    print("output results are saved at: {}".format(args.output_file_name))

    #for line in lines:
    #    eles = line.strip().split(' ')
    #    tt = float(eles[0])
    #    w1 = float(eles[1])
    #    w2 = float(eles[2])
    #    w3 = float(eles[3])
    trainer = Trainer(args, device, args.target)
    trainer.do_training()


if __name__ == "__main__":
    train_with_sweep()
