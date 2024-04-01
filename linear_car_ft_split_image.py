import argparse
from torch import nn
#from data import data_helper
#from optimizer.optimizer_helper import get_optim_and_scheduler
#from utils.Logger import Logger
from datetime import datetime
#from timm.models import create_model
import pickle
#import itertools
#import time
from helpers.image_loader import *
from PIL import Image
from itertools import product
#ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_args():
    parser = argparse.ArgumentParser(description="Makes multiple predictions from a known linear model (e.g. made by promtstyler); joining the predictions to a new one")
    parser.add_argument("--dataset", default="Terra")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of epochs")
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--CLIP", default="ViT-L/14", help="CLIP model")
    parser.add_argument("--output_folder", default='results', help="folder where to save results file")
    parser.add_argument("--data_path", default='../data', help="path of the dataset")
    parser.add_argument("--word_mode", default="mean", help="how the finetuning is computed a) mean [=default]: average all word vectors and finetune the representation b) linear: [=words are more important]: connect all word vectors with a linear layer and finetune those weights")
    parser.add_argument("--KL_factor", default=1)
    return parser.parse_args()


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.autograd.Variable:
        return torch.autograd.Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def tile(filePath, d, overlap_factor=1): #does not generalize yet, # fragments needs to be adapted according to --image_size and self.img_dims
    '''

    :param filePath:
    :param d:
    :param overlap_factor: 2 --> 50% overlap
    :return:
    '''
    img = Image.open(filePath)
    w, h = img.size

    grid = product(range(0, h - h % d, int(np.around(d/overlap_factor, 0))), range(0, w - w % d, int(np.around(d/overlap_factor, 0))))
    fragments = [img]*int(np.around(1+12*overlap_factor**2, 0))
    #print(12*overlap_factor**2)
    for i, j in grid:
        box = (j, i, j + d, i + d)
        #print(int(np.around(4*overlap_factor**2*i, 0))//d + int(np.around(j*overlap_factor, 0))//d +1)
        fragments[int(np.around(4*overlap_factor**2*i, 0))//d + int(np.around(j*overlap_factor, 0))//d +1] = img.crop(box)#.save(out)
        #plt.imshow(img)
        #plt.show()
    return fragments


class Trainer:
    def __init__(self, args, device, tt, ww1, ww2, ww3, target_name):
        self.args = args
        self.device = device

        self.clip_model, self.image_preprocess = clip.load(self.args.CLIP, device=self.device)
        #self.image_preprocess_training = Compose([
        #    torchvision.transforms.RandomResizedCrop(args.image_size, interpolation=BICUBIC),
        #    torchvision.transforms.RandomHorizontalFlip(),
        #    # CenterCrop(args.image_size),
        #    lambda image: image.convert("RGB"),
        #    ToTensor(),
        #    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        #])
        self.clip_frozen, _ = clip.load(self.args.CLIP, device=self.device)
        for name, param in self.clip_frozen.named_parameters():
            param.requires_grad = False

        self.text_feature_dim = 512
        # ---CLIP prompt engineering
        #weighting:
        for iteration, classname in enumerate(args.classes):
            clip_name_loadable = self.args.CLIP.replace("/","")
            try:
                with open("saved_prompts/"+args.dataset+"_"+classname+"_"+clip_name_loadable+".pickle", 'rb') as fp:
                    weights = pickle.load(fp)
                print("loading of "+ "'saved_prompts/"+args.dataset+"_"+classname+"_"+clip_name_loadable+".pickle' successful!" )
            except:
                with (torch.no_grad()):
                    self.clip_model.eval()
                    weights = self.clip_model.encode_text(torch.cat([clip.tokenize(f"a {dID} of a {classname}.") for dID in self.text_anchor]).to(self.device))#.detach().cpu().numpy()
            #cls: 512 x n_cat (W = n_class x n_text_anchor x 512 ; n_class only later "weights"=n_text_anchor x 512
            #ctx: 512 x n_text anchors
                    weights /= weights.norm(dim=-1, keepdim=True)
                    weights = weights.detach().cpu().numpy()

            if iteration == 0:
                weight_total = weights[None, :, :]
            else:
                weight_total = np.concatenate((weight_total, weights[None, :, :]), axis=0)

        with torch.no_grad():
            self.weights_ctx_0 = torch.mean(torch.tensor(weight_total, device=self.device, dtype=torch.float16), axis=0)  # / self.WEIGHTS.size(0)
            self.weights_ctx = self.weights_ctx_0 / self.weights_ctx_0.norm()


        clipper = args.CLIP.replace("/", "")
        self.file_print = open(args.output_folder + "/image_splitter_"+args.word_mode+"_" + clipper + "_" + args.dataset, 'w',
                               encoding="utf-8")
        if args.dataset == "Terra":
            self.img_dims=[1024, 747]
            self.n_splits = 13
            self.n_overlaps = 1
        else:
            print("please define img dims/number splits for the dataset you use [__init__ at like line 8]")
            exit(1)

        #self.clip_model, self.image_preprocess = clip.load(self.args.CLIP, device=self.device)
        #for param in self.clip_model.parameters():
        #    param.requires_grad = False

        clip_name_loadable = self.args.CLIP.replace("/", "")
        with open("saved_prompts/" + args.dataset + "_weights_" + clip_name_loadable + ".pickle", 'rb') as fp:
            self.lin_projection_weights = torch.nn.Parameter(torch.tensor(pickle.load(fp), requires_grad=True, device=self.device, dtype=torch.float16))


        weight_init = np.ones((self.n_splits, 1))
        #for idx in range(len(weight_init)):
        #    weight_init[idx] *= 1 - (idx+1)/self.n_splits
        self.combination_weights = torch.tensor(weight_init, requires_grad=False, device=self.device, dtype=torch.float16)
        self.exponent = 512


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

        self.optimizer = torch.optim.SGD([self.lin_projection_weights ], lr=5e-06*(args.batch_size/64), momentum=0.2, weight_decay=0.1)#5e-03*(args.batch_size/64))
        #self.optimizer = torch.optim.AdamW(params, lr=5*10**-6*(args.batch_size/64), weight_decay=0.1)
        #self.optimizer, self.scheduler = get_optim_and_scheduler(self.model, args.epochs, args.learning_rate, args.train_all,
        #                                                         nesterov=False)
        self.current_epoch = 0

    def _do_epoch(self):
        print("made it to an epoch")
        softmax = nn.Softmax(dim=-1).cuda()
        CELoss = nn.CrossEntropyLoss()
        KLLoss = torch.nn.KLDivLoss(reduction="batchmean")
        cos_sim = torch.nn.CosineSimilarity(dim=-1)
        n_corr = 0
        #for execution_nr in range(self.dataloader.__len__()):  # complex way to write sample ove rall samples
        #_, labels, _, paths = next(iter(self.dataloader))
        n_samples = 0
        for it, (_, class_l, _, paths) in enumerate(self.source_loader):
            class_l = class_l.to(self.device)
            for i, path in enumerate(paths):
                #image = Image.open(path)
                frags = tile(path, d=224, overlap_factor=1)
                #image = torchvision.io.read_image(path)
                for frag_idx, image in enumerate(frags):
                    n_samples+=1
                    data = self.image_preprocess(image).to(self.device).unsqueeze(0)
                    #if i == 0:
                    if frag_idx == 0 and i == 0:
                        data_n = data
                    else:
                        data_n = torch.cat((data_n, data),0)
                    #else:




            CLIP_image_features = self.clip_model.encode_image(data_n).type(torch.float16).to(self.device)
            frozen_image = self.clip_frozen.encode_image(data_n[[i*self.n_splits for i in range(self.args.batch_size)]]).type(torch.float16).to(self.device)
            norm_frozen_image = frozen_image / frozen_image.norm(dim=-1, keepdim=True)
            norm_CLIP_image = (CLIP_image_features[[i*self.n_splits for i in range(self.args.batch_size)]] /
                               CLIP_image_features[[i*self.n_splits for i in range(self.args.batch_size)]]).norm(dim=-1, keepdim=True)

            encodings = (CLIP_image_features @ self.lin_projection_weights.T)
            #frozen_encodings = (frozen_image)

            for iteration in range(len(CLIP_image_features) // self.n_splits):

                sample_idx = 1 + iteration * self.n_splits
                sample_end_idx = self.n_splits + iteration * self.n_splits
                empty_idx = 5 + iteration * self.n_splits
                if self.args.dataset == "Terra":
                    rank_encodings = torch.cat((torch.nn.Softmax(dim=-1)(encodings[sample_idx:sample_end_idx, :empty_idx]),
                                                torch.nn.Softmax(dim=-1)(encodings[sample_idx:sample_end_idx, (empty_idx+1):])), dim=1)
                else:
                    rank_encodings = torch.nn.Softmax(dim=-1)(encodings[1:])
                # Entropy = - \sum_p log_2(p) * p    #probability interpretation makes sense as we used softmax beforea

                entropies = -1* torch.sum(torch.log2(rank_encodings)*rank_encodings, dim=-1)

                _, indices = torch.sort(entropies)
                #print(entropies.size(), encodings.size())
                weighting_factor = (entropies[indices[0]] / entropies[
                    indices, None]) ** self.exponent # -1 * torch.log(entropies[indices[0]] / entropies[
                # indices, None]) * entropies[indices[0]] / entropies[indices, None]

                part_encodings = encodings[sample_idx:sample_end_idx]
                encodings[sample_idx:sample_end_idx] = part_encodings[indices] * weighting_factor
                #frozen_encodings[sample_idx:sample_end_idx] = frozen_encodings[sample_idx:sample_end_idx][indices] * weighting_factor

                normalization = torch.sqrt(1 + torch.sum(weighting_factor ** 2, dim=0))
                encodings[sample_idx:sample_end_idx] /= normalization

                #frozen_encodings[sample_idx:sample_end_idx] /= normalization
            #--mapping to a single output with self.combination_weights
            #print(self.combination_weights.size(), encodings.size())
            #print()
            for sample_idx in range(len(class_l)):
                #print("sample idx:",sample_idx * self.n_splits)
                outputs = (self.combination_weights.T @ encodings[sample_idx*self.n_splits:sample_idx*self.n_splits+self.n_splits])
                if sample_idx == 0:
                    output_list = outputs.type(torch.float16).to(self.device)
                    #f_output_list = frozen_outputs
                else:
                    output_list = torch.cat((output_list,outputs), 0).type(torch.float16).to(self.device)


            #print(encodings.size(), self.weights_ctx.size())
            p_ctx_changing = softmax(norm_CLIP_image[:, None, :] @ self.weights_ctx[None, :, :]).type(torch.float16)
            p_ctx_stationary = softmax((norm_frozen_image[:, None, :] @ self.weights_ctx[None, :, :]).type(torch.float16))
            log_changing = torch.log(p_ctx_changing)
            #data, class_l, d_idx = data.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            torch.autograd.set_detect_anomaly(True)

            #if i == 0:
            #output_list = torch.tensor(outputs)
            #else:
            #    output_list = torch.cat((output_list, outputs), dim=0)
            # Calculate features
            self.clip_model.eval()


            self.optimizer.zero_grad()
            #labels = class_l.repeat_interleave(self.n_splits)
            # --- classification loss
            kl_loss = KLLoss(log_changing, p_ctx_stationary)
            CrossEntropyLoss = CELoss(output_list, class_l)

            loss = CrossEntropyLoss + kl_loss
            loss.backward()
            self.optimizer.step()

            # --- state of training print
            correct_class = torch.argmax(output_list, dim=-1)==class_l
            print("\r", end="")
            n_corr +=  torch.sum(correct_class).cpu().detach().numpy()
            print((it + 1) * len(class_l), " / ", len(self.source_loader.dataset), ": ",
                  np.around(100 * (it + 1) * len(class_l) / len(self.source_loader.dataset), 4), "%  of epoch done.",
                  " Accuracy(batch)=",
                  np.around((100 * torch.sum(correct_class) / len(class_l)).cpu().detach().numpy(), 4), "%",
                  "  Accuracy(total)=",
                  np.around(100 * n_corr / (it + 1) / len(class_l), 4), "%",
                  sep="", end="")

        #
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

        cos_sim = torch.nn.CosineSimilarity(dim=-1)
        class_correct = 0
        n_samples = 0
        for it, (_, class_l, _, paths) in enumerate(loader):
            n_samples += len(class_l)
            class_l = class_l.to(self.device)
            for i, path in enumerate(paths):
                # image = Image.open(path)
                frags = tile(path, d=224, overlap_factor=self.n_overlaps)
                # image = torchvision.io.read_image(path)
                for frag_idx, image in enumerate(frags):
                    data = self.image_preprocess(image).to(self.device).unsqueeze(0)
                    # if i == 0:
                    if frag_idx == 0 and i == 0:
                        data_n = data
                    else:
                        data_n = torch.cat((data_n, data), 0)
                    # else:

            CLIP_image_features = self.clip_model.encode_image(data_n).type(torch.float16).to(self.device)

            encodings = (CLIP_image_features @ self.lin_projection_weights.T)
            # if frag_idx == 0:
            #    encodings = encoding_i
            # else:
            #    encodings = torch.cat((encodings, encoding_i), dim=0)

            # -- sort outputs by entropy --> lowest first index and so on ==> very certain estimates are at low index

            # if terra: dont use empty as much will be empty
            for iteration in range(len(CLIP_image_features) // self.n_splits):

                sample_idx = 1+iteration * self.n_splits
                sample_end_idx = self.n_splits + iteration * self.n_splits
                empty_idx = 5 + iteration * self.n_splits
                if self.args.dataset == "Terra":
                    rank_encodings = torch.cat(
                        (torch.nn.Softmax(dim=-1)(encodings[sample_idx:sample_end_idx, :empty_idx]),
                         torch.nn.Softmax(dim=-1)(encodings[sample_idx:sample_end_idx, (empty_idx + 1):])), dim=1)
                else:
                    rank_encodings = torch.nn.Softmax(dim=-1)(encodings[1:])
                # Entropy = - \sum_p log_2(p) * p    #probability interpretation makes sense as we used softmax beforea

                entropies = -1 * torch.sum(torch.log2(rank_encodings) * rank_encodings, dim=-1)

                _, indices = torch.sort(entropies)
                # print(entropies.size(), encodings.size())
                weighting_factor = (entropies[indices[0]] / entropies[indices, None]) ** self.exponent
                                    #**(2**self.exponent_power_of_2)) #-1 * torch.log(entropies[indices[0]] / entropies[
                    #indices, None]) * entropies[indices[0]] / entropies[indices, None]

                part_encodings = encodings[sample_idx:sample_end_idx]
                encodings[sample_idx:sample_end_idx] = part_encodings[indices] * weighting_factor

                normalization = torch.sqrt(1+ torch.sum(weighting_factor**2, dim=0))
                encodings[sample_idx:sample_end_idx] /= normalization

            # --mapping to a single output with self.combination_weights
            # print(self.combination_weights.size(), encodings.size())
            # print()
            for sample_idx in range(len(class_l)):
                outputs = (self.combination_weights.T @ encodings[
                                                        sample_idx * self.n_splits:sample_idx * self.n_splits + self.n_splits])
                if sample_idx == 0:
                    output_list = outputs
                else:
                    output_list = torch.cat((output_list, outputs), 0)

            predictions = torch.argmax(output_list, axis=-1)
                #predictions = torch.argmax(nn.Softmax(dim=1).cuda()((CLIP_image_features @ self.weights_ctx.T).type(torch.float32)), dim=1)


            class_correct += torch.sum(predictions == class_l)
            print("\r", end="")
            print(str(class_correct.cpu().numpy())+" ouf of", n_samples ,"correct ("+str(class_correct.cpu().numpy()/n_samples*100)+"%)", end="")

        return class_correct

    def do_training(self):
        #self.logger = Logger(self.args, update_frequency=30)
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
        #self.logger.save_best(test_res[idx_best], test_res.max())
        return #self.logger


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
