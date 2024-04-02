import argparse
from torch import nn
import pickle
from helpers.image_loader import *
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_args():
    parser = argparse.ArgumentParser(description="Script to finetune CLIP given a linear layer and text embeddings, does not work properly")
    parser.add_argument("--dataset", default="Terra")
    parser.add_argument("--batch_size", "-b", type=int, default=75, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--CLIP", default="ViT-B/16", help="CLIP model")
    parser.add_argument("--data_path", default='../../data', help="path of the dataset")
    parser.add_argument("--save_clip_model", type=bool, default=False, help="whether to save finetuned clip model")
    parser.add_argument("--norm", type=bool, default=False,
                        help="if to norm image inputs (lin-weights from PS classifier have to fit)")
    return parser.parse_args()


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.clip_model, self.image_preprocess = clip.load(self.args.CLIP, device=self.device)

        # -- get prompt basis (either load or 'a {domain} of a {class}')
        for iteration, classname in enumerate(args.classes):
            clip_name_loadable = self.args.CLIP.replace("/", "")
            try:
                with open("../saved_prompts/" + args.dataset + "_" + classname + "_" + clip_name_loadable + ".pickle",
                          'rb') as fp:
                    text_embed = pickle.load(fp)
                print(
                    "loading of " + "'saved_prompts/" + args.dataset + "_" + classname + "_" + clip_name_loadable + ".pickle' successful!")
            except:
                with (torch.no_grad()):
                    self.clip_model.eval()
                    text_embed = self.clip_model.encode_text(
                        torch.cat([clip.tokenize(f"a {dID} of a {classname}.") for dID in self.args.Domain_ID]).to(
                            self.device))  # .detach().cpu().numpy()
                    if self.args.norm:
                        text_embed /= text_embed.norm(dim=-1, keepdim=True)
                    text_embed = text_embed.detach().cpu().numpy()
            #text embed shape (num_prompts, embedding_size)
            if iteration == 0:
                text_embed_per_class = text_embed[None, :, :]
            else:
                text_embed_per_class = np.concatenate((text_embed_per_class, text_embed[None, :, :]), axis=0)
            #text_embed_per_class shape (n_classes, num_prompts, embedding_size)

        self.embed_tensor = torch.tensor(text_embed_per_class, requires_grad=False, device=self.device,
                                    dtype=torch.float16)



        #-- get linear layer
        clip_name_loadable = self.args.CLIP.replace("/", "")
        with open("../saved_prompts/" + args.dataset + "_weights_" + clip_name_loadable + ".pickle", 'rb') as fp:
            self.lin_projection_weights = torch.tensor(pickle.load(fp), requires_grad=False, device=self.device,
                                                       dtype=torch.float16)

        if args.norm:
            print("using normed encodings")
            self.lin_projection_weights /= self.lin_projection_weights.norm(dim=-1, keepdim=True)

        self.train_data = CheapTestImageDataset(
            base_path=self.args.data_path+"/" + args.dataset,
            domains=args.source, class_names=self.args.classes)
        self.source_loader = torch.utils.data.DataLoader(self.train_data, batch_size=args.batch_size, shuffle=True)

        self.test_data = CheapTestImageDataset(
            base_path=self.args.data_path+"/" + args.dataset,
            domains=args.target, class_names=self.args.classes)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=args.batch_size, shuffle=True)

        self.test_loaders = {"test": self.test_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d,  test %d" % (
                self.train_data.__len__(), self.test_data.__len__()))

        params = [param for param in self.clip_model.parameters()]
        self.optimizer = torch.optim.SGD(params, lr=5e-09*(args.batch_size/64), weight_decay=0.1, momentum=0.8)
        #self.optimizer = torch.optim.AdamW(params, lr=5e-06*(args.batch_size/64), weight_decay=0.1)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 8000, eta_min=0, last_epoch=-1)
        self.current_epoch = 0

    def _do_epoch(self):
        CELoss = nn.CrossEntropyLoss()
        cos_sim = torch.nn.CosineSimilarity(dim=-1)

        n_corr = 0
        n_samples = 0
        for it, (_, class_l, _, paths) in enumerate(self.source_loader):
            class_l = class_l.to(self.device)
            for i, path in enumerate(paths):
                image = Image.open(path)
                data_n = self.image_preprocess(image).to(self.device).unsqueeze(0)
                if i == 0:
                    data = data_n
                else:
                    data = torch.cat((data, data_n), dim=0)

            CLIP_image_features = self.clip_model.encode_image(data).type(torch.float16).to(self.device)
            #cos-sim to prompt-embedding as invariant features
            cs_loss = -1*torch.log(torch.sum(cos_sim(CLIP_image_features[:, None, :], self.embed_tensor[class_l][None,: ,:])))



            # Calculate features
            self.clip_model.eval()
            if self.args.norm:
                CLIP_image_features = CLIP_image_features / CLIP_image_features.norm(dim=-1, keepdim=True)


            encodings = torch.nn.Softmax(dim=-1)(CLIP_image_features @ self.lin_projection_weights.T)
            # --- classification loss
            CrossEntropyLoss = CELoss(encodings, class_l)
            loss = 0.001*CrossEntropyLoss + 1 * cs_loss

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # --- state of training print
            correct_class = torch.argmax(encodings, dim=-1)==class_l
            n_samples += len(class_l)
            n_corr += torch.sum(correct_class).cpu().detach().numpy()
            print("\r", end="")

            print(n_samples," / ", len(self.source_loader.dataset), ": ",
                  np.around(100*n_samples/len(self.source_loader.dataset), 4), "%  of epoch done.",
                  " Accuracy(batch)=",
                  np.around((100*torch.sum(correct_class)/len(class_l)).cpu().detach().numpy(),4),"%",
                  "  Accuracy(total)=",
                  np.around(100*n_corr/n_samples,4), "%",
                  sep="", end="")


        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                print("\n", "--> ", sep="", end="")
                total = len(loader.dataset)
                class_correct = self.do_test(loader)
                class_acc = float(class_correct) / total
                print("Accuracies on "+phase+":", "\t", np.around(100*class_acc, 4),"%", sep="", end="")
                self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):
        class_correct = 0
        total = 0
        top2_corr = 0
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
            if self.args.norm:
                CLIP_image_features /= CLIP_image_features.norm(dim=-1, keepdim=True)

            encodings = torch.nn.Softmax(dim=-1)(CLIP_image_features @ self.lin_projection_weights.T)
            predictions = torch.argmax(encodings, axis=-1)
            top_2_pred = torch.topk(encodings, dim=-1, k=2).indices[:, 1]

            class_correct += torch.sum(predictions == class_l)
            total += len(class_l)
            top2_corr += torch.sum(top_2_pred == class_l) + torch.sum(predictions == class_l)
            print("\r", end="")
            print(str(class_correct.cpu().numpy()) + " ouf of ", total
                  , " correct (" + str(class_correct.cpu().numpy() / total * 100) + "%) ",
                  "(top2: ", np.around(top2_corr.cpu().numpy() / total * 100, 3), "%)", end="", sep="")

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

        if self.args.save_clip_model:
            pref = 'saved_prompts/'
            path_p2 = self.args.dataset + self.args.Domain_ID + self.args.CLIP + ".pickle"
            path = pref + path_p2.replace("/", "")

            with open(path, 'wb') as handle:
                pickle.dump(self.clip_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("CLIP model saved for (test on) ", self.args.Domain_ID ," saved \n", "=======================================", sep="")



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
