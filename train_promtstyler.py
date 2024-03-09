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

from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch CLIP distillation")
    parser.add_argument("--dataset", default="PACS")
    #parser.add_argument("--Domain_ID", default=['sketch', 'photo', 'cartoon', 'art_painting'])
    #parser.add_argument("--classes", default=["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    #parser.add_argument("--image_size", type=int, default=224, help="Image size")
    #parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    #parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    #parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    #parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    #parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscale")
    #parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs for each styleword")
    #parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    #parser.add_argument("--network", default="resnetv2_50x1_bit.goog_in21k_ft_in1k", help="Which network to use")
    #parser.add_argument("--val_size", type=float, default=0.1, help="Validation size (between 0 and 1)")
    #parser.add_argument("--folder_name", default='', help="Used by the logger to save logs")
    #parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--CLIP", default="ViT-B/16", help="CLIP model")
    parser.add_argument("--output_folder", default='ps_orig', help="folder where to save results file")
    #parser.add_argument("--output_file_name", default='.txt', help="results file name")
    parser.add_argument("--data_path", default='', help="path of the dataset")
    parser.add_argument("--number_style_words", "-n", default=80, help="number of stylewords to train")
    parser.add_argument("--style_word_basis", default='a photo style of a',
                        help="wordbasis for which stylewords are created, photo --> pseudoword")
    parser.add_argument("--style_word_index", default=1,
                        help="index of which word in style_word_basis shall be replaced by pseudoword; must be int in [0, len(style_word_basis)]")
    parser.add_argument("--save_style_words", default="no",
                        help='''if 'yes' saves the style-context words as /saved_prompts/[dataset]_[class]_[CLIP model].pickle,
                                if 'extend' extends the style-context words in /saved_prompts/[dataset]_[class]_[CLIP model].pickle by the newly created ones.
                                saves are as numpy arrays''')
    parser.add_argument("--save_lin_weights", default="True",
                        help="if True: save weights for linear mapping in /saved_prompts/[dataset]_weights_[CLIP model].pickle")


    return parser.parse_args()


class WordModel(nn.Module):
    def __init__(self, pseudoCLIP_model, index_to_change=2, word_basis="a photo of a", n_style_words=80, style_word_dim=512, device="cuda", style_words_to_load=None):
        super(WordModel, self).__init__()
        if torch.is_tensor(style_words_to_load):
            self.style_words = style_words_to_load
        else:
            self.style_words = torch.nn.Parameter((torch.randn((n_style_words, style_word_dim))) * 0.001)#0.02 #adding promtstyler
        self.style_words.requires_grad = True
        #for encoding
        self.pseudo_clip_encoder = pseudoCLIP_model
        for name, param in self.pseudo_clip_encoder.named_parameters():
            param.requires_grad = False
        #self.pseudo_clip_encoder.requires_grad = False
        #self.content_words = content_words
        self.index = index_to_change
        self.device = device
        self.basic_phrase = word_basis
        with torch.no_grad():
            self.pseudo_clip_encoder.eval()

            self.style_dummy_token = clip.tokenize(self.basic_phrase).to(self.device).detach()
    def forward(self, content_words, style_index):
        #style_index = 3
        #print("content_words", content_words, "style index", style_index)
        with torch.no_grad():
            style_features = self.pseudo_clip_encoder.encode_text(self.style_dummy_token,
                                                        self.style_words,#[:style_index+1],
                                                        position_pseudo=self.index).to(torch.float32).to(self.device)

        for n_cont, content_word in enumerate(content_words):
            text = self.basic_phrase + " " + content_word
            #print(text)
            sc_token = clip.tokenize(text).to(self.device)
            if n_cont == 0:
                #with torch.no_grad():
                style_content_features = self.pseudo_clip_encoder.encode_text(sc_token,
                                                        self.style_words,#[:style_index+1],
                                                        position_pseudo=self.index).to(torch.float32).to(self.device)
                #if style_index==0:
                #    style_content_features = style_content_features[None, :]
                #else:
                style_content_features = style_content_features[:, None, :]
            else:
                #with torch.no_grad():
                sc_dummy = self.pseudo_clip_encoder.encode_text(sc_token, self.style_words,#[:style_index+1],
                                                                    position_pseudo=self.index).to(torch.float32).to("cuda")
                #if style_index==0:
                #    sc_dummy = sc_dummy[None, :]
                #else:
                sc_dummy = sc_dummy[:, None, :]


                style_content_features = torch.cat(
                    (style_content_features, sc_dummy), 1)
                #print("sc dummy", sc_dummy.size(), "sc_feat", style_content_features.size())

        if torch.sum(torch.isnan(style_features)) >=1:
            print("sc token", sc_token)
            print("used style words", self.style_words[:style_index+1])
            print("style features", style_features)
            print("w.o. to float32", self.pseudo_clip_encoder.encode_text(sc_token, self.style_words[:style_index+1], position_pseudo=self.index, print_intermediate=True))
            torch.save(self.style_words[:style_index+1], "values.pt")
            exit()
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

        norm_i = torch.linalg.norm(style_vec[style_index])
        for j in range(0, style_index):
            #loss += torch.abs(nn.CosineSimilarity)
            loss += torch.abs(1./style_index * style_vec[style_index] @ style_vec[j] /
                     (norm_i * torch.linalg.norm(style_vec[j])))
        return loss

def content_loss(style_content_words, content_words, style_index=1, n_classes=7):
    '''
        loss to maximize cosine similarity with style_content_vector and corresponding content_vector while minimizing all other similarities
        - exp for softmax and log for loss scaling [0,1]-> [infty, 0]
    '''
    #print(style_content_words.size(), content_words.size())
    z = style_content_words[style_index] @ content_words.T / (
            torch.linalg.norm(style_content_words[style_index], axis=-1) * torch.linalg.norm(content_words, axis=-1))
    #print(z.size())
    #exit()
    z = torch.exp(z)
    sum_z_imn = torch.sum(z, dim=-1)

    z_diag = torch.diagonal(z, 0)

    loss = -1. / n_classes* torch.sum( torch.log(z_diag) - torch.log(sum_z_imn) )
    return loss

def style_content_loss(model_output, content_labels, style_index=1, n_classes=7):
    loss = content_loss(model_output[1], content_labels, style_index, n_classes) + style_loss(model_output[0], style_index)
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
        clipper = args.CLIP.replace("/", "")
        self.file_print = open(args.output_folder+"_promtStylerOG_"+clipper+"_"+args.dataset, 'w', encoding="utf-8")
        self.args = args
        self.device = device

        self.clip_model, self.image_preprocess = pseudoCLIP.load(self.args.CLIP, device=self.device)
        if self.args.dataset == "Terra":
            print("please load your finetuned CLIP weight here")
            # model_weights = torch.load("/path/finetuned_clip")
            # self.clip_model.load_state_dict(model_weights)
        self.text_feature_dim = 512  if args.CLIP == "ViT-B/16" else 768#=1024 resnet, =512 Vit-B/16, #768 viT-L/14
        self.n_style_words = self.args.number_style_words


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


        word_model = WordModel(self.clip_model, index_to_change=args.style_word_index+1,
                               n_style_words=self.n_style_words, style_word_dim=self.text_feature_dim,
                               word_basis=args.style_word_basis)
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
        token_style_word_to_class = torch.cat([clip.tokenize(f"{self.args.style_word_basis} {c}.") for c in self.args.classes]).to(
           self.device)
        self.content_features_style_word = self.clip_model.encode_text(token_style_word_to_class).to(torch.float32).to(
           self.device)


        train_noise = True
        #print("init parameter", self.word_model.style_words)
        if train_noise:
            for self.n_style_vec in range(self.n_style_words):
                for self.current_epoch in range(self.args.epochs):
                    m_o = self._do_epoch()

                print("training of style_word #" + str(self.n_style_vec) + " finished")


        final_style_words = self.word_model.style_words.detach().clone()
        token_style_classes = torch.cat(
            [clip.tokenize(f"{self.args.style_word_basis} {c}") for c in self.args.classes]).to(self.device)[:, None, :]

        if self.args.save_style_words == "yes" or self.args.save_style_words == "extend":
            with torch.no_grad():
                for class_token, class_X in zip(token_style_classes, self.args.classes):
                    full_vecs = self.clip_model.encode_text(class_token, final_style_words, position_pseudo=self.args.style_word_index+1).detach().cpu().numpy()

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



        amount_of_vecs_per_class = 1 if self.args.dataset == "Officehome" else 5
        style_content_features = (final_style_words.to(RAM_Device)[:amount_of_vecs_per_class, None, :]*0 + self.content_features_style_word.to(RAM_Device)[None, :, :]) #/ 4.0
        print(token_style_classes.size(), final_style_words.size(), final_style_words[0].size())
        with torch.no_grad():
            sc_features = self.clip_model.encode_text(token_style_classes, final_style_words[0], position_pseudo=self.args.style_word_index+1).to(self.device)
        #print("sc[0]", sc_features[0])

        for i in range(1, final_style_words.size(0)):
            with torch.no_grad():
                sc_features = torch.cat((sc_features, self.clip_model.encode_text(token_style_classes, final_style_words[i], position_pseudo=self.args.style_word_index+1).to(self.device)))
        sc_features = sc_features.to(torch.float32)

        input = sc_features
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
                batch_in = (input[batch_start : batch_start+batchsize])[randperm].to(self.device)
                batch_target = (targets[batch_start : batch_start+batchsize])[randperm].to(self.device)
                lin_model_pred, weights = self.lin_model(batch_in)

                #softmax_pred = arcface_softmax(lin_model_pred, batch_in, weights)
                loss_af = arcface_loss(lin_model_pred, batch_in, weights, batch_target)
                loss = 0 * torch.nn.CrossEntropyLoss()(lin_model_pred, batch_target) + 1 * loss_af#self.lin_model.arcface_loss(lin_model_pred, batch_target)
                print(loss.cpu().detach().numpy(), torch.nn.CrossEntropyLoss()(lin_model_pred, batch_target).cpu().detach().numpy(), sep="; ", end="||")
                self.file_print.write("linear loss:" +str(loss.cpu().detach().numpy()) +" "+ str(torch.nn.CrossEntropyLoss()(lin_model_pred, batch_target).cpu().detach().numpy())+"||")
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

        if self.args.save_lin_weights == "True":
            pref = 'saved_prompts/'
            path_p2 = self.args.dataset + "_weights_" + self.args.CLIP + ".pickle"
            path = pref + path_p2.replace("/", "")

            with open(path, 'wb') as handle:
                pickle.dump(self.lin_model.fc.weight.detach().cpu().numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Linear weights saved \n", "=======================================", sep="")

       #' save_lin_weights
       # ", default="
       # True
       # ",
       #' help = "if True: save weights for linear mapping in /saved_prompts/[dataset]_weights_[CLIP model].pickle"

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
                self.file_print.write("Training Performance: "+ str(class_correct.cpu().detach().numpy())+ " ouf of "+str(intermediate_total) + " correct. (" + str(class_correct.cpu().detach().numpy()/(intermediate_total)*100)+"%) \n")
            print("total accuracy:", 1.0 * class_correct / total)
        # return


def train_with_sweep():
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_num
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
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

    #now = datetime.now().strftime("%m-%d-%y_%H:%M:%S")
    #output_file_name = now + '_' + args.dataset + ".txt"#'_' + args.target + '.txt'
    #output_folder = os.path.join(os.getcwd(), 'results', args.output_folder)
    #if os.path.exists(output_folder):
    #    pass
    #else:
    #    os.makedirs(output_folder)
    #args.output_file_name = os.path.join(output_folder, output_file_name)
    #print("output results are saved at: {}".format(args.output_file_name))

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
