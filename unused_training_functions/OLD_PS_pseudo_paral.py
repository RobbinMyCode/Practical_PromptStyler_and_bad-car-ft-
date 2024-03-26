import clip

from helpers.models import *
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch CLIP distillation")
    parser.add_argument("--dataset", default="PACS")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=15, help="Number of epochs for each styleword")
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--CLIP", default="ViT-B/16", help="CLIP model")
    parser.add_argument("--data_path", default='../../data', help="path of the dataset")
    parser.add_argument("--number_style_words", "-n", default=5, help="number of stylewords to train")
    parser.add_argument("--style_word_basis", default='a photo style of a',
                        help="wordbasis for which stylewords are created, photo --> pseudoword")
    parser.add_argument("--style_word_index", default=1,
                        help="index of which word in style_word_basis shall be replaced by pseudoword; must be int in [0, len(style_word_basis)]")
    parser.add_argument("--save_style_words", default="no",
                        help='''if 'yes' saves the style-context words as /saved_prompts/[dataset]_[class]_[CLIP model].pickle,
                                if 'extend' extends the style-context words in /saved_prompts/[dataset]_[class]_[CLIP model].pickle by the newly created ones.
                                saves are as numpy arrays''')
    parser.add_argument("--save_lin_weights", default="false",
                        help="if True: save weights for linear mapping in /saved_prompts/[dataset]_weights_[CLIP model].pickle")


    return parser.parse_args()

class Trainer:
    def __init__(self, args, device):
        clipper = args.CLIP.replace("/", "")
        self.args = args
        self.device = device

        self.clip_model, self.image_preprocess = pseudoCLIP.load(self.args.CLIP, device=self.device)
        self.text_feature_dim = 512  if args.CLIP == "ViT-B/16" else 768#=1024 resnet, =512 Vit-B/16, #768 viT-L/14
        self.n_style_words = self.args.number_style_words

        token_classes = torch.cat([clip.tokenize(f"{c}") for c in self.args.classes]).to(self.device)
        self.content_features = self.clip_model.encode_text(token_classes).to(torch.float32).to(self.device)
        with torch.no_grad():
            self.clip_model.eval()


        word_model = WordModel(self.clip_model, clip.tokenize, index_to_change=args.style_word_index+1,
                               n_style_words=self.n_style_words, style_word_dim=self.text_feature_dim,
                               word_basis_start=args.style_word_basis)
        self.word_model = word_model.to(self.device)



        self.test_data = CheapTestImageDataset(base_path=self.args.data_path+"/"+args.dataset,
                                    domains=args.target, class_names=self.args.classes)
        self.dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=args.batch_size, shuffle=True)
        self.optimizer_sgd = torch.optim.SGD([self.word_model.style_words],  momentum=0.9, lr=0.002)
        self.lin_epochs = 100

        self.current_epoch = 0
    def _do_epoch(self):
        '''
            trains an epoch for pseudowords
        :return:
        '''
        self.word_model.train()
        self.word_model.zero_grad()

        class_words = self.args.classes
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

        RAM_Device = self.device if (self.args.dataset == "PACS" or self.args.dataset == "VLCS") else "cpu"

        with torch.no_grad():
            sc_features = self.clip_model.encode_text(token_style_classes, final_style_words[0], position_pseudo=self.args.style_word_index+1).to(self.device)

        for i in range(1, final_style_words.size(0)):
            with torch.no_grad():
                sc_features = torch.cat((sc_features, self.clip_model.encode_text(token_style_classes, final_style_words[i], position_pseudo=self.args.style_word_index+1).to(self.device)))
        sc_features = sc_features.to(torch.float32)

        input = sc_features
        targets = torch.tensor(range(len(self.content_features))).repeat(
            len(input)//len(self.content_features), 1, 1).flatten().to(RAM_Device)


        self.input = input
        self.lin_model = Linear(self.text_feature_dim, len(self.content_features)).to(self.device)
        lin_optimizer = torch.optim.SGD(self.lin_model.parameters(), lr=0.005, momentum=0.9)



        batch_size_to_remember = 128 if len(input)>128 else len(input)#32 if self.args.dataset == "Officehome" else 128
        batchsize = batch_size_to_remember
        # lin layer training
        for n_lin_epoch in range(self.lin_epochs):
            for batch_start in range(0, input.size(0), batchsize):
                self.lin_model.train()
                self.lin_model.zero_grad()
                lin_optimizer.zero_grad()

                if batch_start+batchsize > input.size(0):   # end at end of input not beyond
                    batchsize = input.size(0) - batch_start

                randperm = torch.randperm(batchsize)
                batch_in = (input[batch_start : batch_start+batchsize])[randperm].to(self.device)
                batch_target = (targets[batch_start : batch_start+batchsize])[randperm].to(self.device)
                lin_model_pred, weights = self.lin_model(batch_in)


                loss_af = arcface_loss(lin_model_pred, batch_in, weights, batch_target)
                loss = 0 * torch.nn.CrossEntropyLoss()(lin_model_pred, batch_target) + 1 * loss_af

                print(loss.cpu().detach().numpy(), torch.nn.CrossEntropyLoss()(lin_model_pred, batch_target).cpu().detach().numpy(), sep="; ", end="||")
                loss.backward(retain_graph=True)
                lin_optimizer.step()

                batchsize = batch_size_to_remember
                if RAM_Device == "cpu":
                    for itm in [lin_model_pred, weights, batch_in, batch_target]:
                        itm.detach()

            print("epoch "+ str(n_lin_epoch)+ " done.")
        self.lin_model.eval()

        if self.args.save_lin_weights == "True":
            pref = 'saved_prompts/'
            path_p2 = self.args.dataset + "_weights_" + self.args.CLIP + ".pickle"
            path = pref + path_p2.replace("/", "")

            with open(path, 'wb') as handle:
                pickle.dump(self.lin_model.fc.weight.detach().cpu().numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Linear weights saved \n", "=======================================", sep="")

        class_correct = 0
        intermediate_total = 0
        self.clip_model.to(self.device)
        with torch.no_grad():
            for execution_nr in range(self.dataloader.__len__()):
                features, labels, _,  paths = next(iter(self.dataloader))
                intermediate_total += len(labels)
                class_correct += self.do_test(features, labels, paths)
                print("currently ", class_correct.cpu().detach().numpy(), "ouf of ", intermediate_total, " correct. (", class_correct.cpu().detach().numpy()/(intermediate_total)*100,"%)" )
            print("total accuracy:", 1.0 * class_correct / intermediate_total)



def train_with_sweep():
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_num
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:"+args.GPU_num if torch.cuda.is_available() else "cpu")



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

    trainer = Trainer(args, device)
    trainer.do_training()


if __name__ == "__main__":
    train_with_sweep()
