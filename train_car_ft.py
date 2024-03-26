import argparse
from torch import nn
import pickle
from helpers.image_loader import *
import torchvision
from PIL import ImageFile, Image
from torchvision.transforms import Compose, ToTensor, Normalize
ImageFile.LOAD_TRUNCATED_IMAGES = True
from helpers.imagenet_templates import imagenet_templates

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch CAR-FT")
    parser.add_argument("--dataset", default="Terra")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--epochs", "-e", type=int, default=40, help="Number of epochs")
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--CLIP", default="ViT-B/16", help="CLIP model")
    parser.add_argument("--output_folder", default='run1', help="folder where to save results file")
    parser.add_argument("--output_file_name", default='.txt', help="results file name")
    parser.add_argument("--data_path", default='../data', help="path of the dataset")
    parser.add_argument("--prompts_file", default="", help="the pickle-file in which the prompt (already encoded) are [required for weight init], if no file given, use 'a style of a [class]'")
    parser.add_argument("--KL_factor", default=1)
    parser.add_argument("--use_ImageNet_style_words", "-uIw", default=False)
    return parser.parse_args()


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        clipper = args.CLIP.replace("/", "")
        self.file_print = open(args.output_folder + "/car_ft_" + clipper + "_" + args.dataset, 'a',
                               encoding="utf-8")
        self.file_print.write("######################################################################### \n")

        self.clip_model, self.image_preprocess = clip.load(self.args.CLIP, device=self.device)
        self.image_preprocess_training = Compose([
                torchvision.transforms.RandomResizedCrop(args.image_size, interpolation=BICUBIC),
                torchvision.transforms.RandomHorizontalFlip(),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        self.clip_frozen, _ = clip.load(self.args.CLIP, device=self.device)
        for name, param in self.clip_frozen.named_parameters():
            param.requires_grad = False

        #-- get prompt basis (either load, image-net or 'a {domain} of a {class}')
        for iteration, classname in enumerate(args.classes):
            if args.use_ImageNet_style_words:
                templates = imagenet_templates
                texts = [template.format(classname) for template in templates]
                weights = self.clip_model.encode_text(clip.tokenize(texts).to(self.device)).to(
                        self.device)
                weights /= (weights.norm(dim=-1, keepdim=True))
                weights = weights.detach().cpu().numpy()
            else:
                clip_name_loadable = self.args.CLIP.replace("/","")
                try:
                    with open("saved_prompts/"+args.dataset+"_"+classname+"_"+clip_name_loadable+".pickle", 'rb') as fp:
                        weights = pickle.load(fp)
                    print("loading of "+ "'saved_prompts/"+args.dataset+"_"+classname+"_"+clip_name_loadable+".pickle' successful!" )
                except:
                    with (torch.no_grad()):
                        self.clip_model.eval()
                        weights = self.clip_model.encode_text(torch.cat([clip.tokenize(f"a {dID} of a {classname}.") for dID in self.text_anchor]).to(self.device))#.detach().cpu().numpy()
                        weights /= weights.norm(dim=-1, keepdim=True)
                        weights = weights.detach().cpu().numpy()

            if iteration == 0:
                weight_total = weights[None, :, :]
            else:
                weight_total = np.concatenate((weight_total, weights[None, :, :]), axis=0)

        self.WEIGHTS = torch.tensor(weight_total, requires_grad=False, device=self.device, dtype=torch.float16) #torch.nn.Parameter(torch.tensor(weight_total, requires_grad=True, device=self.device, dtype=torch.float16))

        with torch.no_grad():
            self.weights_ctx_0 = torch.mean(self.WEIGHTS, axis=0)
            self.weights_ctx = self.weights_ctx_0 / self.weights_ctx_0.norm()

            self.weights_cls_0 = torch.mean(self.WEIGHTS, axis=1)
            self.weights_cls = torch.nn.Parameter(self.weights_cls_0 / self.weights_cls_0.norm())
        self.weights_cls.requires_grad = True


        self.train_data = CheapTestImageDataset(
            base_path="../data/" + args.dataset,
            domains=args.source, class_names=self.args.classes)
        self.source_loader = torch.utils.data.DataLoader(self.train_data, batch_size=args.batch_size, shuffle=True)

        self.test_data = CheapTestImageDataset(
            base_path="../data/" + args.dataset,
            domains=args.target, class_names=self.args.classes)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=2*args.batch_size, shuffle=True)

        self.test_loaders = {"test": self.test_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d,  test %d" % (
                self.train_data.__len__(), self.test_data.__len__()))

        params = [param for param in self.clip_model.parameters()]+[self.weights_cls]
        self.optimizer = torch.optim.SGD(params, lr=5e-06*(args.batch_size/64), weight_decay=0.1, momentum=0.8)
        #self.optimizer = torch.optim.AdamW(params, lr=5e-06*(args.batch_size/64), weight_decay=0.1)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 8000, eta_min=0, last_epoch=-1)
        self.current_epoch = 0

    def _do_epoch(self):
        softmax = nn.Softmax(dim=-1).cuda()
        CELoss = nn.CrossEntropyLoss()
        KLLoss = torch.nn.KLDivLoss(reduction="batchmean")
        cos_sim = torch.nn.CosineSimilarity(dim=-1)

        n_corr = 0
        for it, (_, class_l, _, paths) in enumerate(self.source_loader):
            class_l = class_l.to(self.device)
            for i, path in enumerate(paths):
                image = Image.open(path)
                data_n = self.image_preprocess_training(image).to(self.device).unsqueeze(0)
                if i == 0:
                    data = data_n
                else:
                    data = torch.cat((data, data_n), dim=0)

            CLIP_image_features = self.clip_model.encode_image(data).type(torch.float16).to(self.device)
            frozen_image = self.clip_frozen.encode_image(data)
            torch.autograd.set_detect_anomaly(True)

            # Calculate features
            self.clip_model.eval()
            CLIP_image_features_norm = CLIP_image_features / CLIP_image_features.norm(dim=-1, keepdim=True)

            with torch.no_grad():
                frozen_image_norm = frozen_image / frozen_image.norm(dim=-1, keepdim=True)

            p_ctx_changing = softmax(CLIP_image_features_norm[:, :] @ self.weights_ctx[:, :].T)
            p_ctx_stationary = softmax(frozen_image_norm[:, :] @ self.weights_ctx[:, :].T).type(torch.float16)
            cls_changing = cos_sim(CLIP_image_features[:, None, :], self.weights_cls[None, :, :]).type(
                torch.float16)

            self.optimizer.zero_grad()

            # --- classification loss
            CrossEntropyLoss = CELoss(cls_changing, class_l)
            # --- kl loss
            log_changing = torch.log(p_ctx_changing)
            kl_loss = KLLoss(log_changing, p_ctx_stationary)

            loss = self.args.KL_factor *kl_loss + CrossEntropyLoss
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

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
                class_correct = self.do_test(loader)
                class_acc = float(class_correct) / total
                print("Accuracies on "+phase+":", "\t", np.around(100*class_acc, 4),"%", sep="", end="")
                self.file_print.write(self.args.target+
                    "__Accuracies on "+phase+":"+ "\t"+ str(np.around(100*class_acc, 4)) + "% \n")
                self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):
        class_correct = 0
        cos_sim = torch.nn.CosineSimilarity(dim=-1)
        for it, (_, class_l, _, paths) in enumerate(loader):
            class_l = class_l.to(self.device)
            for i, path in enumerate(paths):
                data_n = self.image_preprocess(Image.open(path)).to(self.device).unsqueeze(0)
                if i == 0:
                    CLIP_image_features = self.clip_model.encode_image(data_n).type(torch.float16).to(self.device)
                else:
                    CLIP_image_features = torch.cat(
                        (CLIP_image_features,
                         self.clip_model.encode_image(data_n).type(torch.float16).to(self.device)),
                        0)

            CLIP_image_features /= CLIP_image_features.norm(dim=-1, keepdim=True)
            cls_changing = cos_sim(CLIP_image_features[:, None, :], self.weights_cls[None, :, :]).type(
                torch.float16)

            predictions = torch.argmax(cls_changing, axis=-1)
            class_correct += torch.sum(predictions == class_l)

            print("\r", end="")
            print(str(class_correct.cpu().numpy())+" ouf of", (it+1)*len(class_l.cpu().numpy()) ,"correct ("+str(class_correct.cpu().numpy()/((it+1)*len(class_l.cpu().numpy()))*100)+"%)", end="")

        return class_correct

    def do_training(self):
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}

        for self.current_epoch in range(self.args.epochs):
            print("epoch ", self.current_epoch+1, "/", self.args.epochs,": ", sep="")
            self._do_epoch()
            self.optimizer.step()
            print("", end="\n")

        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
        val_res.max(), test_res[idx_best], test_res.max(), idx_best))


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
        args.target = domain
        args.source = args.Domain_ID.copy()
        args.source.remove(args.target)
        print("Training {} on source domains:".format(args.dataset))
        print(*args.source, sep=",")
        print("Test on target domains:")
        print(args.target)

        trainer = Trainer(args, device)
        trainer.do_training()

if __name__ == "__main__":
    train_with_sweep()
