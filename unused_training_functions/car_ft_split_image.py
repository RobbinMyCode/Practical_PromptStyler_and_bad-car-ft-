import argparse
from torch import nn
import pickle
from helpers.image_loader import *
import helpers.imagenet_templates as imagenet_templates
from PIL import Image
from itertools import product


def get_args():
    parser = argparse.ArgumentParser(description="Makes multiple predictions (sub-images) based on CAR-FT; joining the predictions to a new one, KL only on centercrops")
    parser.add_argument("--dataset", default="Terra")
    parser.add_argument("--batch_size", "-b", type=int, default=5, help="Batch size")
    parser.add_argument("--data_path", default='../../data', help="path of the dataset")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of epochs")
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--CLIP", default="ViT-B/16", help="CLIP model")
    parser.add_argument("--KL_factor", default=1)
    parser.add_argument("--use_ImageNet_style_words", "-uIw", default=False)
    parser.add_argument("--swap_entropy", default=2.5,#1.45,
                        help="the value of entropy the best sub-image must be less or equal"
                             " to be swapped with centercrop -->weighted with 1 and centercrop lower")
    return parser.parse_args()


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
    for i, j in grid:
        box = (j, i, j + d, i + d)
        fragments[int(np.around(4*overlap_factor**2*i, 0))//d + int(np.around(j*overlap_factor, 0))//d +1] = img.crop(box)#.save(out)
    return fragments


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        clipper = args.CLIP.replace("/", "")

        self.clip_model, self.image_preprocess = clip.load(self.args.CLIP, device=self.device)
        self.image_preprocess_training = self.image_preprocess
        self.clip_frozen, _ = clip.load(self.args.CLIP, device=self.device)
        for name, param in self.clip_frozen.named_parameters():
            param.requires_grad = False


        #image tiling
        if args.dataset == "Terra":
            self.img_dims=[1024, 747]
            self.n_splits = 13
            self.n_overlaps = 1
            self.norm_bias = 1
        else:
            print("please define img dims/number splits for the dataset used [so far only Terra is defined; __init__ #image tiling]")
            exit(1)

        #combination of sub-images for total encoding: (apart from entropy)
        weight_init = np.ones((self.n_splits, 1))
        self.combination_weights = torch.tensor(weight_init, requires_grad=False, device=self.device,
                                                dtype=torch.float16)

        # -- get prompt basis (either load, image-net or 'a {domain} of a {class}')
        for iteration, classname in enumerate(args.classes):
            if args.use_ImageNet_style_words:
                templates = imagenet_templates
                texts = [template.format(classname) for template in templates]
                weights = self.clip_model.encode_text(clip.tokenize(texts).to(self.device)).to(
                    self.device)
                weights /= (weights.norm(dim=-1, keepdim=True))
                weights = weights.detach().cpu().numpy()
            else:
                clip_name_loadable = self.args.CLIP.replace("/", "")
                try:
                    with open("saved_prompts/" + args.dataset + "_" + classname + "_" + clip_name_loadable + ".pickle",
                              'rb') as fp:
                        weights = pickle.load(fp)
                    print(
                        "loading of " + "'saved_prompts/" + args.dataset + "_" + classname + "_" + clip_name_loadable + ".pickle' successful!")
                except:
                    with (torch.no_grad()):
                        self.clip_model.eval()
                        weights = self.clip_model.encode_text(
                            torch.cat([clip.tokenize(f"a {dID} of a {classname}.") for dID in self.text_anchor]).to(
                                self.device))  # .detach().cpu().numpy()
                        weights /= weights.norm(dim=-1, keepdim=True)
                        weights = weights.detach().cpu().numpy()

            if iteration == 0:
                weight_total = weights[None, :, :]
            else:
                weight_total = np.concatenate((weight_total, weights[None, :, :]), axis=0)

        self.WEIGHTS = torch.tensor(weight_total, requires_grad=False, device=self.device,
                                    dtype=torch.float16)  # torch.nn.Parameter(torch.tensor(weight_total, requires_grad=True, device=self.device, dtype=torch.float16))

        with torch.no_grad():
            self.weights_ctx_0 = torch.mean(self.WEIGHTS, axis=0)
            self.weights_ctx = self.weights_ctx_0 / self.weights_ctx_0.norm()

            self.weights_cls_0 = torch.mean(self.WEIGHTS, axis=1)
            self.weights_cls = torch.nn.Parameter(self.weights_cls_0 / self.weights_cls_0.norm())
        self.weights_cls.requires_grad = True

        self.train_data = CheapTestImageDataset(
            base_path=self.args.data_path + "/" + args.dataset,
            domains=args.source, class_names=self.args.classes)
        self.source_loader = torch.utils.data.DataLoader(self.train_data, batch_size=args.batch_size, shuffle=True)

        self.test_data = CheapTestImageDataset(
            base_path=self.args.data_path + "/" + args.dataset,
            domains=args.target, class_names=self.args.classes)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=10 * args.batch_size, shuffle=True)

        self.test_loaders = {"test": self.test_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d,  test %d" % (
            self.train_data.__len__(), self.test_data.__len__()))

        params = [param for param in self.clip_model.parameters()] + [self.weights_cls]
        self.optimizer = torch.optim.SGD(params, lr=5e-06 * (args.batch_size / 64), weight_decay=0.1, momentum=0.8)
        # self.optimizer = torch.optim.AdamW(params, lr=5e-06*(args.batch_size/64), weight_decay=0.1)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 8000, eta_min=0, last_epoch=-1)
        self.current_epoch = 0

    def _do_epoch(self):
        softmax = nn.Softmax(dim=-1).cuda()
        CELoss = nn.CrossEntropyLoss()
        KLLoss = torch.nn.KLDivLoss(reduction="batchmean")
        cos_sim = torch.nn.CosineSimilarity(dim=-1)

        n_corr = 0
        n_samples = 0

        for it, (_, class_l, _, paths) in enumerate(self.source_loader):
            class_l = class_l.to(self.device)
            for i, path in enumerate(paths):
                frags = tile(path, d=224, overlap_factor=self.n_overlaps)
                for frag_idx, image in enumerate(frags):
                    n_samples += 1
                    data = self.image_preprocess(image).to(self.device).unsqueeze(0)
                    if frag_idx == 0 and i == 0:
                        data_n = data
                        f_data_n = data
                    else:
                        data_n = torch.cat((data_n, data), 0)
                        if frag_idx == 0:
                            f_data_n = torch.cat((f_data_n, data), 0)

            CLIP_image_features = self.clip_model.encode_image(data_n).type(torch.float16).to(self.device)


            cls_changing = cos_sim(CLIP_image_features[:, None, :], self.weights_cls[None, :, :]).type(
                torch.float16)

            encodings = cls_changing

            # -- sort outputs by entropy --> lowest first index and so on ==> very certain estimates are at low index
            # if terra: dont use empty as much will be empty
            for iteration in range(len(CLIP_image_features) // self.n_splits):
                sample_idx = 1 + iteration * self.n_splits
                sample_end_idx = self.n_splits + iteration * self.n_splits
                empty_idx = 5 + iteration * self.n_splits

                if self.args.dataset == "Terra":
                    rank_encodings = torch.cat(
                        (torch.nn.Softmax(dim=-1)(encodings[sample_idx:sample_end_idx, :empty_idx]),
                         torch.nn.Softmax(dim=-1)(encodings[sample_idx:sample_end_idx, (empty_idx + 1):])), dim=1)
                    # all probs sum up to 1 (without "empty") --> important for entropy
                    rank_encodings /= torch.sum(rank_encodings, dim=0)
                else:
                    rank_encodings = torch.nn.Softmax(dim=-1)(encodings[1:])

                # Entropy = - \sum_p log_2(p) * p    #probability interpretation makes sense as we used softmax beforea
                entropies = -1 * torch.sum(torch.log2(rank_encodings) * rank_encodings, dim=-1)

                _, indices = torch.sort(entropies)
                if entropies[indices[0]] < self.args.swap_entropy:
                    # compare entropy of centercrop
                    encoding_0 = torch.cat(
                        (torch.nn.Softmax(dim=-1)(encodings[sample_idx - 1, :empty_idx]),
                         torch.nn.Softmax(dim=-1)(encodings[sample_idx - 1, (empty_idx + 1):])), dim=-1)
                    encoding_0 /= torch.sum(encoding_0, dim=0)
                    entropy_0 = -1 * torch.sum(torch.log2(encoding_0) * encoding_0, dim=-1)
                    # only swap if min entropy isnt centercrop
                    if entropies[indices[0]] < entropy_0:
                        encodings[sample_idx - 1], encodings[indices[0]] = encodings[indices[0]], encodings[
                            sample_idx - 1]

                # weights are still computed wrt to the best sample, ignoring the swap
                weighting_factor = (entropies[indices[0]] / entropies[indices, None]) ** 512

                part_encodings = encodings[sample_idx:sample_end_idx]
                encodings[sample_idx:sample_end_idx] = part_encodings[indices] * weighting_factor

                normalization = torch.sqrt(self.norm_bias + torch.sum(weighting_factor ** 2, dim=0))
                encodings[sample_idx:sample_end_idx] /= normalization

            # --mapping to a single output with self.combination_weights
            for sample_idx in range(len(class_l)):
                outputs = (self.combination_weights.T @ encodings[
                                                        sample_idx * self.n_splits:sample_idx * self.n_splits + self.n_splits])
                if sample_idx == 0:
                    output_list = outputs
                else:
                    output_list = torch.cat((output_list, outputs), 0)


            #KL PART
            frozen_image = self.clip_frozen.encode_image(f_data_n).type(torch.float16).to(self.device)
            with torch.no_grad():
                frozen_image_norm = frozen_image / frozen_image.norm(dim=-1, keepdim=True)
            CLIP_image_features = self.clip_model.encode_image(f_data_n).type(torch.float16).to(self.device)
            CLIP_image_features_norm = CLIP_image_features / CLIP_image_features.norm(dim=-1, keepdim=True)

            p_ctx_changing = softmax(CLIP_image_features_norm[:, :] @ self.weights_ctx[:, :].T)
            p_ctx_stationary = softmax(frozen_image_norm[:, :] @ self.weights_ctx[:, :].T).type(torch.float16)
            log_changing = torch.log(p_ctx_changing)
            kl_loss = KLLoss(log_changing, p_ctx_stationary)


            torch.autograd.set_detect_anomaly(True)
            self.clip_model.eval()

            self.optimizer.zero_grad()
            CrossEntropyLoss = CELoss(output_list, class_l)

            loss = CrossEntropyLoss + self.args.KL_factor * kl_loss
            loss.backward()
            self.optimizer.step()

            # --- state of training print
            correct_class = torch.argmax(output_list, dim=-1) == class_l
            print("\r", end="")
            n_corr += torch.sum(correct_class).cpu().detach().numpy()
            print((it + 1) * len(class_l), " / ", len(self.source_loader.dataset), ": ",
                  np.around(100 * (it + 1) * len(class_l) / len(self.source_loader.dataset), 4),
                  "%  of epoch done.",
                  " Accuracy(batch)=",
                  np.around((100 * torch.sum(correct_class) / len(class_l)).cpu().detach().numpy(), 4), "%",
                  "  Accuracy(total)=",
                  np.around(100 * n_corr / (it + 1) / len(class_l), 4), "%",
                  sep="", end="")

        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                print("\n", " ", sep="", end="")
                total = len(loader.dataset)
                class_correct = self.do_test(loader)
                class_acc = float(class_correct) / total
                print("Accuracies on " + phase + ":", "\t", np.around(100 * class_acc, 4), "%", sep="", end="")
                self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):
        '''
            gives accuracy and top2 accuracy for the data in the laoder
        :param loader: which laoder to be used --> which data to evaluate
        :return: #correctly classified samples (top1)
        '''
        cos_sim = torch.nn.CosineSimilarity(dim=-1)
        class_correct = 0
        n_samples = 0
        top2_corr = 0
        for it, (_, class_l, _, paths) in enumerate(loader):
            n_samples += len(class_l)
            class_l = class_l.to(self.device)
            for i, path in enumerate(paths):
                frags = tile(path, d=224, overlap_factor=self.n_overlaps)

                for frag_idx, image in enumerate(frags):
                    data = self.image_preprocess(image).to(self.device).unsqueeze(0)

                    if frag_idx == 0 and i == 0:
                        data_n = data
                    else:
                        data_n = torch.cat((data_n, data), 0)

            CLIP_image_features = self.clip_model.encode_image(data_n).type(torch.float16).to(self.device)
            CLIP_image_features /= CLIP_image_features.norm(dim=-1, keepdim=True)

            cls_changing = cos_sim(CLIP_image_features[:, None, :], self.weights_cls[None, :, :]).type(
                torch.float16)
            encodings = cls_changing

            # -- sort outputs by entropy --> lowest first index and so on ==> very certain estimates are at low index
            # if terra: dont use empty as much will be empty
            for iteration in range(len(CLIP_image_features) // self.n_splits):

                sample_idx = iteration * self.n_splits + 1
                sample_end_idx = self.n_splits + iteration * self.n_splits
                empty_idx = 5 + iteration * self.n_splits
                if self.args.dataset == "Terra":
                    rank_encodings = torch.cat(
                        (torch.nn.Softmax(dim=-1)(encodings[sample_idx:sample_end_idx, :empty_idx]),
                         torch.nn.Softmax(dim=-1)(encodings[sample_idx:sample_end_idx, (empty_idx + 1):])), dim=1)
                    # -- normalization without empty class
                    rank_encodings /= torch.sum(rank_encodings, dim=0)
                else:
                    rank_encodings = torch.nn.Softmax(dim=-1)(encodings[1:])
                # Entropy = - \sum_p log_2(p) * p    #probability interpretation makes sense as we used softmax beforea

                entropies = -1 * torch.sum(torch.log2(rank_encodings) * rank_encodings, dim=-1)
                _, indices = torch.sort(entropies)

                if entropies[indices[0]] < self.args.swap_entropy:
                    # compare entropy of centercrop
                    encoding_0 = torch.cat(
                        (torch.nn.Softmax(dim=-1)(encodings[sample_idx - 1, :empty_idx]),
                         torch.nn.Softmax(dim=-1)(encodings[sample_idx - 1, (empty_idx + 1):])), dim=-1)
                    encoding_0 /= torch.sum(encoding_0, dim=0)
                    entropy_0 = -1 * torch.sum(torch.log2(encoding_0) * encoding_0, dim=-1)
                    # only swap if min entropy isnt centercrop
                    if entropies[indices[0]] < entropy_0:
                        encodings[sample_idx - 1], encodings[indices[0]] = encodings[indices[0]], encodings[
                            sample_idx - 1]

                weighting_factor = (entropies[indices[0]] / entropies[indices, None]) ** 512

                part_encodings = encodings[sample_idx:sample_end_idx]
                encodings[sample_idx:sample_end_idx] = part_encodings[indices] * weighting_factor

                normalization = torch.sqrt(self.norm_bias + torch.sum(weighting_factor ** 2, dim=0))
                encodings[sample_idx:sample_end_idx] /= normalization

            #if self.args.norm:
            #    encodings /= encodings.norm(dim=-1, keepdim=True)

            # --mapping to a single output with self.combination_weights
            for sample_idx in range(len(class_l)):
                outputs = (self.combination_weights.T @
                           encodings[sample_idx * self.n_splits:sample_idx * self.n_splits + self.n_splits])
                if sample_idx == 0:
                    output_list = outputs
                else:
                    output_list = torch.cat((output_list, outputs), 0)

            predictions = torch.argmax(output_list, dim=-1)
            top_2_pred = torch.topk(output_list, dim=-1, k=2).indices[:, 1]
            print_corr_example = False
            if print_corr_example:
                for i in range(0, len(predictions)):
                    if predictions[i] == class_l[i]:
                        print(i, paths[i], "correct!")
                        frags = tile(paths[i], d=224, overlap_factor=self.n_overlaps)
                        for idx, tile_x in enumerate(frags):
                            tile_x.save("/home/robin/Documents/Domain_Generalization/test/img" + str(idx) + ".png")

                        from torchvision.transforms import CenterCrop
                        crop_image = CenterCrop(224)(frags[0])
                        crop_image.save("/home/robin/Documents/Domain_Generalization/test/img_cCrop.png")
                        exit()
            class_correct += torch.sum(predictions == class_l)
            top2_corr += torch.sum(top_2_pred == class_l) + torch.sum(predictions == class_l)
            print("\r", end="")
            print(str(class_correct.cpu().numpy()) + " ouf of ", n_samples
                  , " correct (" + str(class_correct.cpu().numpy() / n_samples * 100) + "%) ",
                  "(top2: ", np.around(top2_corr.cpu().numpy() / n_samples * 100, 3), "%)", end="", sep="")

        return class_correct


    def do_training(self):
            self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
            for self.current_epoch in range(self.args.epochs):
                print("epoch ", self.current_epoch+1, "/", self.args.epochs,": ", sep="")
                self._do_epoch()
                print("", end="\n")

            val_res = self.results["val"]
            test_res = self.results["test"]
            idx_best = val_res.argmax()
            print("Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
            val_res.max(), test_res[idx_best], test_res.max(), idx_best))

            return


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
