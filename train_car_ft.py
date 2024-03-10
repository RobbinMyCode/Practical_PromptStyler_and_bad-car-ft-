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
from timm.models import create_model
import pickle
import itertools
import numpy as np
import time
from image_loader import CheapTestImageDataset
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch CAR-FT")
    parser.add_argument("--dataset", default="Terra")
    parser.add_argument("--Domain_ID", default=['sketch', 'photo', 'cartoon', 'art_painting'])
    parser.add_argument("--classes", default=["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--min_scale", default=0.3, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.0, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.0, type=float, help="Chance of randomly greyscale")
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", default="resnetv2_50x1_bit.goog_in21k_ft_in1k", help="Which network to use")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='', help="Used by the logger to save logs")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--CLIP", default="ViT-B/16", help="CLIP model")
    parser.add_argument("--output_folder", default='run1', help="folder where to save results file")
    parser.add_argument("--output_file_name", default='.txt', help="results file name")
    parser.add_argument("--data_path", default='../data', help="path of the dataset")
    parser.add_argument("--prompts_file", default="", help="the pickle-file in which the prompt (already encoded) are [required for weight init], if no file given, use 'a style of a [class]'")
    parser.add_argument("--word_mode", default="mean", help="how the finetuning is computed a) mean [=default]: average all word vectors and finetune the representation b) linear: [=words are more important]: connect all word vectors with a linear layer and finetune those weights")
    return parser.parse_args()


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.autograd.Variable:
        return torch.autograd.Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
class Trainer:
    def __init__(self, args, device, tt, ww1, ww2, ww3, target_name):
        self.args = args
        self.device = device

        clipper = args.CLIP.replace("/", "")
        self.file_print = open(args.output_folder + "car_ft_"+args.word_mode+"_" + clipper + "_" + args.dataset, 'w',
                               encoding="utf-8")

        self.clip_model, self.image_preprocess = clip.load(self.args.CLIP, device=self.device)
        self.clip_frozen, _ = clip.load(self.args.CLIP, device=self.device)
        for name, param in self.clip_frozen.named_parameters():
            param.requires_grad = False

        self.text_feature_dim = 512
        # ---CLIP prompt engineering


        if args.dataset == "Terra":
            self.text_anchor = ['bright photo', 'corrupted photo', 'dark photo', 'good photo']
        elif args.dataset == "VLCS":
            self.text_anchor = ['bright photo', 'corrupted photo', 'dark photo', 'good photo']
        else:
            self.text_anchor = args.Domain_ID


        for iteration, classname in enumerate(args.classes):
            clip_name_loadable = self.args.CLIP.replace("/","")
            try:
                with open("saved_prompts/"+args.dataset+"_"+classname+"_"+clip_name_loadable+".pickle", 'rb') as fp:
                    weights = pickle.load(fp)
                print("loading of "+ "'saved_prompts/"+args.dataset+"_"+classname+"_"+clip_name_loadable+".pickle' successful!" )
            except:
                with torch.no_grad():
                    self.clip_model.eval()
                    weights = self.clip_model.encode_text(torch.cat([clip.tokenize(f"a {dID} of a {classname}.") for dID in self.text_anchor]).to(self.device)).detach().cpu().numpy()
            #cls: 512 x n_cat (W = n_class x n_text_anchor x 512 ; n_class only later "weights"=n_text_anchor x 512
            #ctx: 512 x n_text anchors
            if iteration == 0:
                weight_total = weights[None, :, :]
            else:
                weight_total = np.concatenate((weight_total, weights[None, :, :]), axis=0)

        self.WEIGHTS = torch.nn.Parameter(torch.tensor(weight_total, requires_grad=True, device=self.device, dtype=torch.float16))
        if args.word_mode == "linear":
            self.linear_weight_cls = torch.nn.Parameter(1. / weight_total.shape[1] * torch.ones((weight_total.shape[1], 1), device=self.device,
                                                                    dtype=torch.float16, requires_grad=True))
            self.linear_weight_ctx = torch.nn.Parameter(1. / weight_total.shape[0] * torch.ones((weight_total.shape[0], 1) , device=self.device,
                                                                    dtype=torch.float16, requires_grad=True))
            self.WEIGHTS.requires_grad = False

        self.train_data = CheapTestImageDataset(
            base_path="../data/" + args.dataset,
            domains=args.source, class_names=self.args.classes)
        self.source_loader = torch.utils.data.DataLoader(self.train_data, batch_size=args.batch_size, shuffle=True)

        self.test_data = CheapTestImageDataset(
            base_path="../data/" + args.dataset,
            domains=args.target, class_names=self.args.classes)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=args.batch_size, shuffle=True)


        #self.source_loader, self.val_loader = data_helper.get_train_dataloader(args)
        #self.target_loader = data_helper.get_val_dataloader(args)
        self.test_loaders = {"test": self.test_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d,  test %d" % (
                self.train_data.__len__(), self.test_data.__len__()))

        params = [param for param in self.clip_model.parameters()]+[self.WEIGHTS]
        self.optimizer = torch.optim.SGD(params, lr=5e-06*(args.batch_size/64), weight_decay=0.1)
        #self.optimizer = torch.optim.AdamW(params, lr=5*10**-6*(args.batch_size/64), weight_decay=0.1)
        #self.optimizer, self.scheduler = get_optim_and_scheduler(self.model, args.epochs, args.learning_rate, args.train_all,
        #                                                         nesterov=False)
        self.current_epoch = 0

    def _do_epoch(self):
        softmax = nn.Softmax(dim=-1).cuda()
        CELoss = nn.CrossEntropyLoss()
        cos_sim = torch.nn.CosineSimilarity(dim=-1)

        n_corr = 0
        #for execution_nr in range(self.dataloader.__len__()):  # complex way to write sample ove rall samples
        #_, labels, _, paths = next(iter(self.dataloader))
        for it, (_, class_l, _, paths) in enumerate(self.source_loader):
            class_l = class_l.to(self.device)
            for i, path in enumerate(paths):
                data_n = self.image_preprocess(Image.open(path)).to(self.device).unsqueeze(0)
                if i == 0:
                    CLIP_image_features = self.clip_model.encode_image(data_n).type(torch.float32).to(self.device)
                    frozen_image = self.clip_frozen.encode_image(data_n)
                else:
                    CLIP_image_features = torch.cat(
                        (CLIP_image_features, self.clip_model.encode_image(data_n).type(torch.float32).to(self.device)),
                        0)
                    frozen_image = torch.cat(
                        (frozen_image, self.clip_frozen.encode_image(data_n).type(torch.float32).to(self.device)),
                        0)

            if self.args.word_mode == "mean":
                self.weights_cls = torch.sum(self.WEIGHTS, axis=1) / self.WEIGHTS.size(1)
                self.weights_ctx = torch.sum(self.WEIGHTS, axis=0) / self.WEIGHTS.size(0)

            if self.args.word_mode == "linear":
                self.weights_cls = torch.squeeze(self.WEIGHTS.permute(0, 2, 1) @ self.linear_weight_cls)
                self.weights_ctx = torch.squeeze(self.WEIGHTS.permute(1, 2, 0) @ self.linear_weight_ctx)

            #data, class_l, d_idx = data.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            torch.autograd.set_detect_anomaly(True)

            # Calculate features
            self.clip_model.eval()
            #CLIP_image_features = self.clip_model.encode_image(data)
            CLIP_image_features_norm = CLIP_image_features / CLIP_image_features.norm(dim=-1, keepdim=True)

            with torch.no_grad():
                #frozen_image = self.clip_frozen.encode_image(data)
                frozen_image_norm = frozen_image / frozen_image.norm(dim=-1, keepdim=True)

            p_ctx_changing = softmax(cos_sim(CLIP_image_features_norm[:, None, :], self.weights_ctx[None, :, :]))
            p_ctx_stationary = softmax((cos_sim(frozen_image_norm[:, None, :], self.weights_ctx[None, :, :])).type(torch.float32))
            cls_changing = cos_sim(CLIP_image_features_norm[:, None, :], self.weights_cls[None, :, :]).type(torch.float32)

            self.optimizer.zero_grad()

            # --- classification loss
            CrossEntropyLoss = CELoss(cls_changing, class_l)
            # --- kl loss
            log_changing = torch.log(p_ctx_changing)
            log_stat  = torch.log(p_ctx_stationary)
            kl_loss = F.kl_div(log_stat,
                               log_changing,
                               log_target=True,
                               reduction='batchmean')

            loss = kl_loss + CrossEntropyLoss
            loss.backward()
            self.optimizer.step()

            # --- state of training print
            correct_class = torch.argmax(cls_changing, dim=-1)==class_l
            print("\r", end="")
            n_corr +=  torch.sum(correct_class).cpu().detach().numpy()
            print((it+1)*len(class_l)," / ", len(self.source_loader.dataset), ": ",
                  np.around(100*(it+1)*len(class_l)/len(self.source_loader.dataset), 4), "%  of epoch done.",
                  " Accuracy(batch)=",
                  np.around((100*torch.sum(correct_class)/len(class_l)).cpu().detach().numpy(),4),"%",
                  "  Accuracy(total)=",
                  np.around(100*n_corr/(it+1)/len(class_l),4), "%",
                  sep="", end="")


        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                print("\n", "--> ", sep="", end="")
                total = len(loader.dataset)
                #print("TOTAL", total)
                class_correct = self.do_test(loader)
                #print("CLASS CORRECT", class_correct)
                class_acc = float(class_correct) / total
                print("Accuracies on "+phase+":", "\t", np.around(100*class_acc, 4),"%", sep="", end="")
                self.file_print.write(self.args.target+
                    "__Accuracies on "+phase+":"+ "\t"+ str(np.around(100*class_acc, 4)) + "% \n")
                #self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][self.current_epoch] = class_acc
        #print("========================================================")
        #print("phase and loader done")
        #print("========================================================")
    def do_test(self, loader):
        class_correct = 0
        cos_sim = torch.nn.CosineSimilarity(dim=-1)
        for it, (_, class_l, _, paths) in enumerate(loader):
            class_l = class_l.to(self.device)
            for i, path in enumerate(paths):
                data_n = self.image_preprocess(Image.open(path)).to(self.device).unsqueeze(0)
                if i == 0:
                    CLIP_image_features = self.clip_model.encode_image(data_n).type(torch.float32).to(self.device)
                else:
                    CLIP_image_features = torch.cat(
                        (CLIP_image_features,
                         self.clip_model.encode_image(data_n).type(torch.float32).to(self.device)),
                        0)

            #data, class_l = data.to(self.device), class_l.to(self.device)
            #CLIP_image_features = self.clip_model.encode_image(data)
            CLIP_image_features /= CLIP_image_features.norm(dim=-1, keepdim=True)

            if self.args.word_mode == "mean":
                weights_cls = torch.sum(self.WEIGHTS, axis=1) / self.WEIGHTS.size(1)

            if self.args.word_mode == "linear":
                weights_cls = torch.squeeze(self.WEIGHTS.permute(0, 2, 1) @ self.linear_weight_cls)

            cls_changing = cos_sim(CLIP_image_features[:, None, :], weights_cls[None, :, :]).type(
                torch.float32)
            predictions = torch.argmax(cls_changing, axis=-1)
                #predictions = torch.argmax(nn.Softmax(dim=1).cuda()((CLIP_image_features @ self.weights_ctx.T).type(torch.float32)), dim=1)


            class_correct += torch.sum(predictions == class_l)
            print("\r", end="")
            print(str(class_correct.cpu().numpy())+" ouf of", (it+1)*len(class_l.cpu().numpy()) ,"correct ("+str(class_correct.cpu().numpy()/((it+1)*len(class_l.cpu().numpy()))*100)+"%)", end="")

        return class_correct

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            #self.logger.new_epoch(self.scheduler.get_last_lr())
            print("epoch ", self.current_epoch+1, "/", self.args.epochs,": ", sep="")
            self._do_epoch()
            print("", end="\n")
            #self.scheduler.step()
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
        val_res.max(), test_res[idx_best], test_res.max(), idx_best))
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger


def train_with_sweep():
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_num
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:"+args.GPU_num if torch.cuda.is_available() else "cpu")


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

    for domain in args.Domain_ID:
        #if domain == "location_100" or domain == "location_38" or domain == "location_43":
        #    continue
        args.target = domain
        args.source = args.Domain_ID.copy()
        args.source.remove(args.target)
        print("Training {} on source domains:".format(args.dataset))
        print(*args.source, sep=",")
        print("Test on target domains:")
        print(args.target)

        now = datetime.now().strftime("%m-%d-%y_%H:%M:%S")
        output_file_name = now + '_' + args.dataset + '_' + args.target + '.txt'
        output_folder = os.path.join(os.getcwd(), 'results', args.output_folder)
        if os.path.exists(output_folder):
            pass
        else:
            os.makedirs(output_folder)
        args.output_file_name = os.path.join(output_folder, output_file_name)
        print("output results are saved at: {}".format(args.output_file_name))


        trainer = Trainer(args, device, 0, 0, 0, 0, args.target)
        trainer.do_training()

if __name__ == "__main__":
    train_with_sweep()
