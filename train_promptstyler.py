import open_clip
from helpers.models import *
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch PromptStyler (OG) training")
    parser.add_argument("--dataset", default="Terra")
    parser.add_argument("--batch_size", "-b", type=int, default=500, help="Batch size (to load images with)")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs for each styleword")
    parser.add_argument("--lin_epochs", "-le", type=int, default=50, help="Number of epochs for each styleword")
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--CLIP", default="ViT-B/16", help="CLIP model")
    parser.add_argument("--output_folder", default='results', help="folder where to save training results file")
    parser.add_argument("--data_path", default='../data', help="path of the dataset")
    parser.add_argument("--number_style_words", "-n", default=80, help="number of stylewords to train")
    parser.add_argument("--style_word_basis", default='a wildlife camera recording of a',
                        help="wordbasis for which stylewords are created, photo --> pseudoword")
    parser.add_argument("--style_word_index", default=4,
                        help="index of which word in style_word_basis shall be replaced by pseudoword; must be int in [0, len(style_word_basis)]")
    parser.add_argument("--save_style_words", default="yes",
                        help='''if 'yes' saves the style-context words as /saved_prompts/[dataset]_[class]_[CLIP model].pickle,
                                if 'extend' extends the style-context words in /saved_prompts/[dataset]_[class]_[CLIP model].pickle by the newly created ones.
                                saves are as numpy arrays''')
    parser.add_argument("--save_lin_weights", type=bool, default=True,
                        help="if True: save weights for linear mapping in /saved_prompts/[dataset]_weights_[CLIP model].pickle")
    parser.add_argument("--norm", default=False, help="if to norm text and image embeddings")


    return parser.parse_args()




class Trainer:
    def __init__(self, args, device):
        clipper = args.CLIP.replace("/", "")
        self.file_print = open(args.output_folder+"/PromptStylerOG_"+clipper+"_"+args.dataset, 'a', encoding="utf-8")
        self.args = args
        self.device = device

        # -- clip model (1st clip vs 2nd open_clip)
        self.clip_model, self.image_preprocess = pseudoCLIP.load(self.args.CLIP, device=self.device)
        self.text_feature_dim = 512 if args.CLIP == "ViT-B/16" else 768  # =1024 resnet, =512 Vit-B/16, #768 viT-L/14
        self.tokenizer = clip.tokenize
        self.use_large_models = False
        if args.CLIP == "ViT-H/14" or args.CLIP == "ViT-h/14" or args.CLIP == "ViT-G/14" or args.CLIP == "ViT-g/14":
            self.text_feature_dim = 1024
            CLIP_tokenizer = args.CLIP.replace("/", "-")
            self.tokenizer = open_clip.get_tokenizer(CLIP_tokenizer)
            self.use_large_models = True


        self.n_style_words = self.args.number_style_words
        token_classes = torch.cat([self.tokenizer(f"{c}") for c in self.args.classes]).to(self.device)
        self.content_features = self.clip_model.encode_text(token_classes).to(torch.float32).to(self.device)
        with torch.no_grad():
            self.clip_model.eval()


        word_model_static = WordModel(self.clip_model, self.tokenizer, index_to_change=args.style_word_index+1,
                               n_style_words=self.n_style_words, style_word_dim=self.text_feature_dim,
                               word_basis_start=args.style_word_basis)
        self.word_model_static = word_model_static.to(self.device)
        self.word_model_static.style_words.requires_grad = False


        self.test_data = CheapTestImageDataset(base_path=self.args.data_path+"/"+ args.dataset,
                                    domains=args.target, class_names=self.args.classes)
        self.dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=args.batch_size, shuffle=True)

        self.current_epoch = 0
    def _do_epoch(self):
        '''
            trains an epoch for pseudowords
        :return:
        '''
        self.word_model_dyn.train()
        self.word_model_dyn.zero_grad()

        class_words = self.args.classes
        self.optimizer_sgd.zero_grad()
        model_output = self.word_model_dyn(class_words)
        with torch.no_grad():
            total_output = self.word_model_static(class_words)
        #print("before:", total_output[0][ :5, :5], "\n")
        total_output[0][self.n_style_vec] = model_output[0]
        total_output[1][self.n_style_vec] = model_output[1]

        word_loss = style_content_loss(total_output, self.content_features,
                                       style_index=self.n_style_vec, n_classes=len(self.args.classes))

        if self.current_epoch == self.args.epochs and self.n_style_vec == self.n_style_words -1:
            word_loss.backward(retain_graph=False)
        else:
            word_loss.backward(retain_graph=True)

        self.optimizer_sgd.step()
        self.word_model_dyn.eval()

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
        if self.args.norm:
            CLIP_image_features /= CLIP_image_features.norm(dim=-1, keepdim=True)

        predictions, _ = self.lin_model(CLIP_image_features)
        softmax_pred = torch.nn.Softmax(dim=-1)(predictions)

        pred_index = torch.argmax(softmax_pred, axis=-1)
        class_correct += torch.sum(pred_index == class_l)
        return class_correct

    def do_training(self):
        token_style_word_to_class = torch.cat([self.tokenizer(f"{self.args.style_word_basis} {c}.") for c in self.args.classes]).to(
           self.device)
        self.content_features_style_word = self.clip_model.encode_text(token_style_word_to_class).to(torch.float32).to(
           self.device)


        for self.n_style_vec in range(self.n_style_words):
            word_model_dynamic = WordModel(self.clip_model, self.tokenizer, index_to_change=self.args.style_word_index + 1,
                                            n_style_words=1, style_word_dim=self.text_feature_dim,
                                            word_basis_start=self.args.style_word_basis,
                                            style_words_to_load= self.word_model_static.style_words[self.n_style_vec][None,:])
            self.word_model_dyn = word_model_dynamic.to(self.device)
            self.optimizer_sgd = torch.optim.SGD([self.word_model_dyn.style_words], momentum=0.9, lr=0.002)
            for self.current_epoch in range(self.args.epochs):
                m_o = self._do_epoch()

            print("training of style_word #" + str(self.n_style_vec) + " finished")
            with torch.no_grad():
                self.word_model_static.style_words[self.n_style_vec] = self.word_model_dyn.style_words.detach().clone()

        final_style_content_words = self.word_model_static(self.args.classes)[1].detach().cpu().numpy()

        # --save style words

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

            # preparation for lin layer
        RAM_Device = self.device if (self.args.dataset == "PACS" or self.args.dataset == "VLCS") else "cpu"
        self.input = torch.flatten(torch.tensor(final_style_content_words).to(RAM_Device), end_dim=1)


        if self.args.norm:
            self.input /= self.input.norm(dim=-1, keepdim=True)

        targets = torch.tensor(range(len(self.content_features))).repeat( #1*len(flattened_sc_no)+
            len(self.input)//len(self.content_features), 1, 1).flatten().to(RAM_Device) # permute(1,0)



        self.lin_model = Linear(self.text_feature_dim, len(self.content_features)).to(self.device)
        lin_optimizer = torch.optim.SGD(self.lin_model.parameters(), lr=0.005, momentum=0.9)


        #lin layer training
        batch_size_to_remember = 128 if len(self.input)>128 else len(self.input)#32 if self.args.dataset == "Officehome" else 128
        batchsize = batch_size_to_remember


        for n_lin_epoch in range(self.args.lin_epochs):
            for batch_start in range(0, self.input.size(0), batchsize):
                self.lin_model.train()
                self.lin_model.zero_grad()
                lin_optimizer.zero_grad()

                if batch_start+batchsize > self.input.size(0):   # end at end of input not beyond
                    batchsize = self.input.size(0) - batch_start
                #change order of words in lin layer training
                randperm = torch.randperm(batchsize)
                batch_in = (self.input[batch_start : batch_start+batchsize])[randperm].to(self.device)
                batch_target = (targets[batch_start : batch_start+batchsize])[randperm].to(self.device)

                lin_model_pred, weights = self.lin_model(batch_in)
                loss_af = arcface_loss(lin_model_pred, batch_in, weights, batch_target)
                loss = 0 * torch.nn.CrossEntropyLoss()(lin_model_pred, batch_target) + 1 * loss_af

                print(loss.cpu().detach().numpy(), torch.nn.CrossEntropyLoss()(lin_model_pred, batch_target).cpu().detach().numpy(), sep="; ", end="||")
                self.file_print.write("linear loss:" +str(loss.cpu().detach().numpy()) +" "+ str(torch.nn.CrossEntropyLoss()(lin_model_pred, batch_target).cpu().detach().numpy())+"||")
                loss.backward(retain_graph=True)
                lin_optimizer.step()
                batchsize = batch_size_to_remember

                #-- reduce size on gpu for large num of classes
                if RAM_Device == "cpu":
                    for itm in [lin_model_pred, weights, batch_in, batch_target]:
                        itm.detach()

            print("epoch "+ str(n_lin_epoch)+ " done.")
        self.lin_model.eval()

        #--save linear layer
        if self.args.save_lin_weights:
            pref = 'saved_prompts/'
            path_p2 = self.args.dataset + "_weights_" + self.args.CLIP + ".pickle"
            path = pref + path_p2.replace("/", "")

            with open(path, 'wb') as handle:
                pickle.dump(self.lin_model.fc.weight.detach().cpu().numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Linear weights saved \n", "=======================================", sep="")


        #--evaluate on images
        class_correct = 0
        intermediate_total = 0
        self.clip_model.to(self.device)
        with torch.no_grad():
            for execution_nr in range(self.dataloader.__len__()):  # complex way to write sample ove rall samples
                features, labels, _,  paths = next(iter(self.dataloader))
                intermediate_total += len(labels)
                class_correct += self.do_test(features, labels, paths)
                print("currently ", class_correct.cpu().detach().numpy(), "ouf of ", intermediate_total, " correct. (", class_correct.cpu().detach().numpy()/(intermediate_total)*100,"%)" )
                self.file_print.write("Training Performance: "+ str(class_correct.cpu().detach().numpy())+ " ouf of "+str(intermediate_total) + " correct. (" + str(class_correct.cpu().detach().numpy()/(intermediate_total)*100)+"%) \n")
            print("total accuracy:", 100. * class_correct / intermediate_total)


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
