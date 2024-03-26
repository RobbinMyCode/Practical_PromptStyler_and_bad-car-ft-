import clip

from helpers.models import *
def get_args():
    parser = argparse.ArgumentParser(description="Script to launch CLIP distillation")
    parser.add_argument("--dataset", default="Terra")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of epochs for each styleword")
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--CLIP", default="ViT-B/16", help="CLIP model")
    parser.add_argument("--data_path", default='', help="path of the dataset")
    parser.add_argument("--number_style_words", "-n", default=2, help="number of stylewords to train")
    parser.add_argument("--style_word_basis", default='a photo of a',
                        help="wordbasis for which stylewords are created, photo --> pseudoword")
    parser.add_argument("--style_word_index", default=1,
                        help="index of which word in style_word_basis shall be replaced by pseudoword; must be int in [0, len(style_word_basis)]")
    parser.add_argument("--save_style_words", default="no",
                        help='''if 'yes' saves the style-context words as /saved_prompts/[dataset]_[class]_[CLIP model].pickle,
                                if 'extend' extends the style-context words in /saved_prompts/[dataset]_[class]_[CLIP model].pickle by the newly created ones.
                                saves are as numpy arrays''')
    parser.add_argument("--save_lin_weights", default="False",
                        help="if True: save weights for linear mapping in /saved_prompts/[dataset]_weights_[CLIP model].pickle")
    return parser.parse_args()


class Trainer:
    def __init__(self, args, device, target_name):
        self.args = args
        self.device = device

        self.clip_model, self.image_preprocess = pseudoCLIP.load(self.args.CLIP, device=self.device)
        self.text_feature_dim = 512  if args.CLIP == "ViT-B/16" else 768#=1024 resnet, =512 Vit-B/16, #768 viT-L/14
        self.n_style_words = self.args.number_style_words


        token_classes = torch.cat([clip.tokenize(f"{c}") for c in self.args.classes]).to(self.device)
        self.content_features = self.clip_model.encode_text(token_classes).to(torch.float32).to(self.device)
        with torch.no_grad():
            self.clip_model.eval()

        self.target_name = target_name
        self.test_data = CheapTestImageDataset(
            base_path="/home/robin/Documents/Domain_Generalization/data/" + args.dataset,
            domains=self.args.target, class_names=self.args.classes)
        self.dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=args.batch_size, shuffle=True)

    def _do_epoch(self):
        '''
            trains an epoch for pseudowords
        :return:
        '''
        self.word_model.train()
        self.word_model.zero_grad()

        class_words = self.args.classes #torch.cat((self.args.classes, self.content_features_style_word), 0)[torch.randint(len(self.args.classes), (128,))]
        self.optimizer_sgd.zero_grad()
        model_output = self.word_model(class_words)
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
        if self.args.dataset == "Terra":
            word_basises_start = ["a photo of a", "a nice photo of a", "a photo of the large", "a wildlife camera recording of a"] # #a wildlife [photography] of
            word_basises_end = ["", "", "", ""]
            pseudo_index = [2 ,2, 5, 4]
        else:
            word_basises_start = [self.args.style_word_basis]
            word_basises_end = [""]
            pseudo_index = [self.args.style_word_index+1]

        first_run = True
        for change_index, wb_start, wb_end in zip(pseudo_index, word_basises_start, word_basises_end):
            print("==========================   pseudo basis: " + wb_start + " [class] " +
                  wb_end +" ; pseudo at index", change_index-1, " =====================================")

            word_model = WordModel(self.clip_model, clip.tokenize, index_to_change=change_index,
                                   n_style_words=self.n_style_words, style_word_dim=self.text_feature_dim,
                                   word_basis_start=wb_start, word_basis_end=wb_end)
            self.word_model = word_model.to(self.device)


            self.optimizer_sgd = torch.optim.SGD(word_model.parameters(), momentum=0.9, lr=0.002)
            self.lin_epochs = 100

            self.current_epoch = 0


            token_style_word_to_class = torch.cat([clip.tokenize(f"{self.args.style_word_basis} {c}.") for c in self.args.classes]).to(
               self.device)
            self.content_features_style_word = self.clip_model.encode_text(token_style_word_to_class).to(torch.float32).to(
               self.device)

            for self.n_style_vec in range(self.n_style_words):
                for self.current_epoch in range(self.args.epochs):
                    m_o = self._do_epoch()
                print(" -- training of style_word #" + str(self.n_style_vec+1) + " finished.")

            final_style_content_dummy = self.word_model(self.args.classes)[1].detach().cpu().numpy()
            if first_run:
                final_style_content_words = final_style_content_dummy
            else:
                final_style_content_words = torch.cat(final_style_content_words, final_style_content_dummy)

        if self.args.save_style_words == "yes" or self.args.save_style_words == "extend":
            with torch.no_grad():
                for idx, class_X in enumerate(self.args.classes):
                    full_vecs = final_style_content_words[:, idx, :]

                    pref = 'saved_prompts/'
                    path_p2 = self.args.dataset + "_" + class_X + "_" + self.args.CLIP + ".pickle"
                    path = pref + path_p2.replace("/", "")

                    if self.args.save_style_words == "extend":
                        with open(path, 'rb') as handle:  # with statement --> auto close
                            old_vecs = pickle.load(handle)
                        full_vecs = np.concatenate((old_vecs, full_vecs), axis=0)

                    with open(path, 'wb') as handle:
                        pickle.dump(full_vecs, handle, protocol=pickle.HIGHEST_PROTOCOL)

                print("Word vectors saved \n", "=======================================", sep="")

        RAM_Device = self.device if (self.args.dataset == "PACS" or self.args.dataset == "VLCS") else "cpu"
        input = torch.flatten(torch.tensor(final_style_content_words).to(RAM_Device), end_dim=1)
        targets = torch.tensor(range(len(self.content_features))).repeat(
            len(input)//len(self.content_features), 1, 1).flatten().to(RAM_Device)


        self.input = input
        self.lin_model = Linear(self.text_feature_dim, len(self.content_features)).to(self.device)
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

    args.target = args.Domain_ID
    print("Test on target domains (images):")
    print(args.target)
    print("Classes are:", args.classes)

    trainer = Trainer(args, device, args.target)
    trainer.do_training()


if __name__ == "__main__":
    train_with_sweep()
