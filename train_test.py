import os
import argparse
import torch
import clip
from torch import nn
from torch.nn import functional as F
from data import data_helper
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
from datetime import datetime
from pytorch_metric_learning import losses
from image_loader import CheapTestImageDataset
import itertools
from timm.models import create_model

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
    parser.add_argument("--val_size", type=float, default=1.0, help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='', help="Used by the logger to save logs")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--CLIP", default="ViT-B/16", help="CLIP model")
    parser.add_argument("--output_folder", default='run1', help="folder where to save results file")
    parser.add_argument("--output_file_name", default='.txt', help="results file name")
    parser.add_argument("--data_path", default='', help="path of the dataset")

    return parser.parse_args()


class WordModel(nn.Module):
    def __init__(self, a_embedding, style_of_a_embedding, n_style_words=80, style_word_dim=512, style_words_to_load = None):
        super(WordModel, self).__init__()
        if torch.is_tensor(style_words_to_load):
            self.style_words = style_words_to_load
        #self.style_words = torch.nn.Parameter(torch.normal(0, 0.02, size=(n_style_words, style_word_dim) ))
        else:
            self.style_words = torch.nn.Parameter((torch.randn((n_style_words, style_word_dim)) ) * 0.02) #og promtstyler
            #self.style_words = torch.nn.Parameter((torch.randn((n_style_words, style_word_dim))) * 0.5) #adding promtstyler
        self.a_embed = a_embedding
        self.style_of_a_embed = style_of_a_embedding
        for x in [self.a_embed, self.style_of_a_embed]:
            x.required_grad = False
    def forward(self, class_words):
        style_features = (self.a_embed + self.style_words + self.style_of_a_embed) #/ 3.0
        style_content_features = (self.a_embed + self.style_words[:, None, :] + self.style_of_a_embed + class_words[None, :, :]) #/ 4.0
        #style_content_features = self.style_words[:, None, :] + class_words[None, :, :]
        return [style_features, style_content_features]

class ArcFaceLinear(nn.Module):
    def __init__(self, dim_in, dim_out, s=5, m=0.5):
        super().__init__()
        #for loss func:
        self.sin_m = torch.sin(torch.tensor(m))
        self.cos_m = torch.cos(torch.tensor(m))
        self.s = s
        self.dim_out = dim_out

        #for classifier
        self.weight_tensor = torch.nn.Parameter((torch.randn((dim_in, dim_out))) * 0.1) #init gaussian
        self.normed_weight_tensor = self.weight_tensor.detach().clone() #becomes normed weight tensor in forward
    def forward(self, style_content_words):
        norm_words = torch.sqrt(torch.sum(style_content_words**2, axis=-1)).unsqueeze(-1)
        normed_content_words = style_content_words/norm_words

        norm_weights = torch.sqrt(torch.sum(self.weight_tensor**2, axis=0)).unsqueeze(0)
        normed_weights = self.weight_tensor / norm_weights
        self.normed_weight_tensor = normed_weights.detach().clone()

        cos_theta = normed_content_words @ normed_weights # x and w normed --> = cos_theta
        return cos_theta

    def arcface_loss(self, cos_theta, labels):
        '''
        :param model:
        :param cos_theta: contains ALL angles wrt output (batchsize,outdim)
        :param labels: labebls as numerical values (batchsize) [1..outdim]
        :return:
        '''

        sin_theta = (1 - cos_theta ** 2) ** 0.5
        cos_theta_plus_m = cos_theta * self.cos_m - sin_theta * self.sin_m #sin_theta+m

        exp_pos = torch.exp(self.s * cos_theta_plus_m)
        exp_neg = torch.exp(self.s * cos_theta)

        one_hot_labels = F.one_hot(labels, num_classes=self.dim_out)
        numerator = torch.max(exp_pos*one_hot_labels[None, :], axis=-1)[0].flatten()
        denominator = torch.sum(torch.abs(1 - one_hot_labels) * exp_neg, axis=-1) +numerator

        L3 = - torch.log(numerator/denominator) #loss for each sample
        return torch.sum(L3)
    def arcface_softmax(self, cos_theta):
        sin_theta = (1 - cos_theta ** 2) ** 0.5
        cos_theta_plus_m = cos_theta * self.cos_m - sin_theta * self.sin_m  # sin_theta+m

        exp_pos = torch.exp(self.s * cos_theta_plus_m)
        exp_neg = torch.exp(self.s * cos_theta)

        #one_hot_labels = F.one_hot(labels, num_classes=self.dim_out)
        numerator = exp_pos #torch.max(exp_pos * one_hot_labels[None, :], axis=-1)[0].flatten()
        denominator_sum_all = torch.sum(exp_neg, axis=-1)
        result = numerator
        for i in range(result.size(1)):
            result[:, i] /= (denominator_sum_all - exp_neg[:, i] + exp_pos[:, i])

        return result

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
    z = style_content_words[style_index] @ content_words.T / (
            torch.linalg.norm(style_content_words[style_index], axis=-1) * torch.linalg.norm(content_words, axis=-1))
    z = torch.exp(z)
    sum_z_imn = torch.sum(z, dim=-1)
    z_diag = torch.diagonal(z, 0)

    loss = -1. / n_classes* torch.sum( torch.log(z_diag) - torch.log(sum_z_imn) )
    return loss

def style_content_loss(model_output, content_labels, style_index=1, n_classes=7):
    loss = content_loss(model_output[1], content_labels, style_index, n_classes) + style_loss(model_output[0], style_index)
    return loss

class Trainer:
    def __init__(self, args, device, target_name):
        self.args = args
        self.device = device

        self.clip_model, _ = clip.load(self.args.CLIP, device=self.device)
        if self.args.dataset == "Terra":
            print("please load your finetuned CLIP weight here")
            # model_weights = torch.load("/path/finetuned_clip")
            # self.clip_model.load_state_dict(model_weights)

        # ---CLIP prompt engineering

        self.num_style_vectors = 80


        self.text_feature_dim = 512  #=1024 resnet, =512 Vit-B/16, #768 viT-L/14
        style_word_dim = 512  # resnet, vit-b, #768 vit-l
        self.n_style_words = 80


        if args.dataset == "Terra":
            self.text_anchor = ['bright photo', 'corrupted photo', 'dark photo', 'good photo']
        elif args.dataset == "VLCS":
            self.text_anchor = ['bright photo', 'corrupted photo', 'dark photo', 'good photo']
        else:
            self.text_anchor = args.source  ###all named styles except the one for testing e.g. could be PACS: ['art_painting', 'sketch', 'photo'] but not 'cartoon'

        token_a = clip.tokenize(f"a ").to(self.device)
        self.style_words = torch.normal(0, 0.02, size=(self.n_style_words, style_word_dim), device=self.device)

        token_style_of_a = clip.tokenize(f" style of a ").to(self.device)
        token_classes = torch.cat([clip.tokenize(f"{c}") for c in self.args.classes]).to(self.device)


        self.a_embedding = self.clip_model.encode_text(token_a)
        self.a_style_of_a_embedding = self.clip_model.encode_text(token_style_of_a)

        self.content_features = self.clip_model.encode_text(token_classes).to(torch.float32).to(self.device)
        with torch.no_grad():
            self.clip_model.eval()


        word_model = WordModel(self.a_embedding, self.a_style_of_a_embedding,
                               self.n_style_words, self.text_feature_dim)
        self.word_model = word_model.to(self.device)

        self.test_data = CheapTestImageDataset(base_path="/home/robin/Documents/Domain_Generalization/data/"+args.dataset,
                                    domains=args.target, class_names=self.args.classes)
        self.dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=args.batch_size, shuffle=True)
        self.optimizer_sgd = torch.optim.SGD(word_model.parameters(), lr=0.002, momentum=0.9)
        self.lin_epochs = 50

        self.current_epoch = 0
        self.target_name = target_name
    def _do_epoch(self):
        self.word_model.train()
        self.word_model.zero_grad()

        class_vec_samples = self.content_features[torch.randint(len(self.args.classes), (128,))]
        self.optimizer_sgd.zero_grad()
        model_output = self.word_model(class_vec_samples)
        word_loss = style_content_loss(model_output, class_vec_samples,
                                       style_index=self.n_style_vec, n_classes=len(self.args.classes))

        if self.current_epoch == self.args.epochs and self.n_style_vec == self.n_style_words -1:
            word_loss.backward(retain_graph=False)
        else:
            word_loss.backward(retain_graph=True)
        self.optimizer_sgd.step()  # step works
        self.word_model.eval()
        print('|', end='', sep='')

    def do_test(self, features, labels):
        import numpy as np
        class_correct = 0
        data, class_l = features.to(self.device), labels.to(self.device)
        #print(data, class_l)
        CLIP_image_features = self.clip_model.encode_image(data).type(torch.float32)
        #print("image feature size",CLIP_image_features.size())
        predictions = self.lin_model(CLIP_image_features)
        softmax_pred = self.lin_model.arcface_softmax(predictions)
        pred_index = torch.argmax(softmax_pred, axis=-1)
        #print(predictions[:10], "\n ========== ", softmax_pred[:10] ,"\n ==========", "pred index", pred_index[:10], "===")
        #print(class_l[:10])
            #student_embedding, student_logits = self.model(data)
            #similarity = student_logits.softmax(dim=-1)
            #if self.args.dataset == "Terra":
            #    student_embedding /= student_embedding.norm(dim=-1, keepdim=True)
            #    student_logits_clip = (100.0 * student_embedding @ self.text_features_ems.T.type(torch.float32)).type(torch.float32)
            #    similarity_clip = student_logits_clip.softmax(dim=-1)
            #    similarity += similarity_clip
            #_, cls_pred = similarity.max(dim=1)
        class_correct += torch.sum(pred_index == class_l)
        return class_correct

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}

        #print("Parameters", [p for p in self.word_model.parameters()])
       # for self.n_style_vec in range(self.n_style_words):
       #     for self.current_epoch in range(self.args.epochs):
       #         #print("epoch ", self.current_epoch)
       #         self._do_epoch()
       #     print("training of style_word #" + str(self.n_style_vec) + " finished")
        #print("========== \n ========== \n" + "After training: \n", self.word_model.style_words)

        final_style_words = self.word_model.style_words.detach().clone()#*0.00001
        #self.a_style_of_a_embedding = 0.000001
        style_content_features = (self.a_embedding + final_style_words[:, None, :] +
                                  self.a_style_of_a_embedding + self.content_features[None, :, :]) #/ 4.0
        #style_content_features = ((torch.randn((80, 512)) ) * 0.000000001).to(self.device)[:, None, :] + self.content_features[None, :, :]
        flattened_sc_features = torch.flatten(style_content_features, end_dim=1)

        input = (flattened_sc_features).to(self.device)#.permute(1,0)
        #for i in range(4):
        #    print(input[5*i:5*(i+1)])
        targets = torch.tensor(range(len(self.content_features))).repeat(
            len(final_style_words), 1).flatten().to(self.device) # permute(1,0)

        self.input = input
        self.lin_model = ArcFaceLinear(self.text_feature_dim, len(self.content_features)).to(self.device)
        lin_optimizer = torch.optim.SGD(self.lin_model.parameters(), lr=0.005, momentum=0.9)
        #print(input.size(), targets.size())

        #lin layer training
        batchsize = 128
        for n_lin_epoch in range(self.lin_epochs):
            for batch_start in range(0, input.size(0), batchsize):
                self.lin_model.train()
                self.lin_model.zero_grad()
                lin_optimizer.zero_grad()
                #self.optimizer_ArcFace.zero_grad()
                if batch_start+batchsize > input.size(0):   # end at end of input not beyond
                    batchsize = input.size(0) - batch_start

                torch.autograd.set_detect_anomaly(True)
                randperm = torch.randperm(batchsize)
                batch_in = (input[batch_start : batch_start+batchsize])[randperm]
                batch_target = (targets[batch_start : batch_start+batchsize])[randperm]
                #print("batch_in", batch_in, batch_start, batchsize)
                lin_model_pred = self.lin_model(batch_in)
                #print("wighs init \n", self.lin_model.fc1_normed.weight_v, self.lin_model.fc1_normed.weight_g)
                loss = self.lin_model.arcface_loss(lin_model_pred, batch_target)
                print(loss.cpu().detach().numpy(), end="||")
                loss.backward(retain_graph=True)
                #print("wighs beefore \n", self.lin_model.fc1_normed.weight_v, self.lin_model.fc1_normed.weight_g)
                #print("grads \n", self.lin_model.fc1_normed.weight_v.grad, self.lin_model.fc1_normed.weight_g.grad)
                lin_optimizer.step()
                #print("wighs after \n", self.lin_model.fc1_normed.weight_v, self.lin_model.fc1_normed.weight_g)
                #self.lin_model.eval()
                #exit("done")
                batchsize = 128
            print("epoch "+ str(n_lin_epoch)+ " done.")
        self.lin_model.eval()

        class_correct = 0
        with torch.no_grad():
            for execution_nr in range(self.dataloader.__len__()):  # complex way to write sample ove rall samples
                features, labels, _ = next(iter(self.dataloader))
                total = self.test_data.__len__()
                class_correct += self.do_test(features, labels)
                print("currently ", class_correct, "ouf of ", (execution_nr + 1) * self.args.batch_size, " correct.")
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
        args.Domain_ID = ["LABELME", "SUN", "VOC", "CALTECH"]
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
                        "Toothbrush", "Toys", "Trash_Can", "TV", "Webcam"]
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
