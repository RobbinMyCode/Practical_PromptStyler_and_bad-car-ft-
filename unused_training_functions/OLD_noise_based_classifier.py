import os
import argparse
import torch
import clip
from torch import nn
from helpers.image_loader import CheapTestImageDataset
from helpers.losses import style_content_loss, arcface_loss
from helpers.models import Linear

def get_args():
    parser = argparse.ArgumentParser(description="Trains a pure natural Prompt Classifier (with noises on [class] in addition)")
    parser.add_argument("--dataset", default="PACS")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs for each styleword")
    parser.add_argument("--data_path", default='../../data', help="path of the dataset")
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--CLIP", default="ViT-B/16", help="CLIP model")
    return parser.parse_args()


class WordModel(nn.Module):
    def __init__(self, n_style_words=80, style_word_dim=512, style_words_to_load = None):
        super(WordModel, self).__init__()
        if torch.is_tensor(style_words_to_load):
            self.style_words = style_words_to_load
        else:
            self.style_words = torch.nn.Parameter((torch.randn((n_style_words, style_word_dim))) * 0.02) #adding promtstyler
    def forward(self, class_words):
        style_features_no___ = self.style_words
        style_content_features___ = self.style_words[:, None, :] + class_words[None, :, :]
        return [style_features_no___, style_content_features___]


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.clip_model, self.image_preprocess = clip.load(self.args.CLIP, device=self.device)
        self.text_feature_dim = 512  if args.CLIP == "ViT-B/16" else 768#=1024 resnet, =512 Vit-B/16, #768 viT-L/14
        self.n_style_words = 14


        token_classes = torch.cat([clip.tokenize(f"{c}") for c in self.args.classes]).to(self.device)
        self.content_features = self.clip_model.encode_text(token_classes).to(torch.float32).to(self.device)
        with torch.no_grad():
            self.clip_model.eval()


        word_model = WordModel(self.n_style_words, self.text_feature_dim)
        self.word_model = word_model.to(self.device)

        self.test_data = CheapTestImageDataset(base_path=self.args.data_path+"/"+args.dataset,
                                    domains=args.target, class_names=self.args.classes)
        self.dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=args.batch_size, shuffle=True)
        self.optimizer_sgd = torch.optim.SGD(word_model.parameters(), lr=0.002, momentum=0.9)
        self.lin_epochs = 15

        self.current_epoch = 0
    def _do_epoch(self):
        '''
            trains an epoch for "pseudowords": noise for classes
        :return:
        '''
        self.word_model.train()
        self.word_model.zero_grad()

        class_vec_samples = torch.cat((self.content_features, self.content_features_style_word), 0)[torch.randint(len(self.args.classes), (128,))]
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
        token_style_word_to_class = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in self.args.classes]).to(
            self.device)
        self.content_features_style_word = self.clip_model.encode_text(token_style_word_to_class).to(torch.float32).to(
            self.device)

        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}

        train_noise = True
        if train_noise:
            for self.n_style_vec in range(self.n_style_words):
                for self.current_epoch in range(self.args.epochs):
                    self._do_epoch()
                print("training of style_word #" + str(self.n_style_vec) + " finished")

        final_style_words = self.word_model.style_words.detach().clone()#
        final_style_words_no = self.word_model.style_words.detach().clone()

        RAM_Device = self.device if (self.args.dataset == "PACS" or self.args.dataset == "VLCS") else "cpu"

        t1 = torch.cat([clip.tokenize(f"itap of a {c}.") for c in self.args.classes]).to(RAM_Device)
        t2 = torch.cat([clip.tokenize(f"a bad photo of the {c}.") for c in self.args.classes]).to(RAM_Device)
        t3 = torch.cat([clip.tokenize(f"a origami {c}.") for c in self.args.classes]).to(RAM_Device)
        t4 = torch.cat([clip.tokenize(f"a photo of the large {c}.") for c in self.args.classes]).to(RAM_Device)
        t5 = torch.cat([clip.tokenize(f"a {c} in a video game.") for c in self.args.classes]).to(RAM_Device)
        t6 = torch.cat([clip.tokenize(f"art of the {c}.") for c in self.args.classes]).to(RAM_Device)
        t7 = torch.cat([clip.tokenize(f"a photo of the small {c}.") for c in self.args.classes]).to(RAM_Device)

        s1 = torch.cat([clip.tokenize(f"a drawing of a {c}.") for c in self.args.classes]).to(RAM_Device)
        s2 = torch.cat([clip.tokenize(f"a sketch of a {c}.") for c in self.args.classes]).to(RAM_Device)
        s3 = torch.cat([clip.tokenize(f"an image of a {c}.") for c in self.args.classes]).to(RAM_Device)

        s4 = torch.cat([clip.tokenize(f"a painting of a {c}.") for c in self.args.classes]).to(RAM_Device)
        s5 = torch.cat([clip.tokenize(f"a cartoon of a {c}.") for c in self.args.classes]).to(RAM_Device)
        s6 = torch.cat([clip.tokenize(f"a cartoon {c}.") for c in self.args.classes]).to(RAM_Device)


        amount_of_vecs_per_class = 1 if self.args.dataset == "Officehome" else 5
        style_content_features = (final_style_words.to(RAM_Device)[:amount_of_vecs_per_class, None, :]*0 + self.content_features_style_word.to(RAM_Device)[None, :, :]) #/ 4.0


        for t in [t1,t2,t3,t4,t5,t6,t7, s1,s2,s3, s4, s5, s6]:
            style_content_features = torch.cat((style_content_features, final_style_words.to(RAM_Device)[:amount_of_vecs_per_class, None, :] * 0 +
                                                self.clip_model.encode_text(t.to(self.device)).to(torch.float32).to(RAM_Device)[None, :, :]), 0)

        style_content_features = torch.cat((style_content_features, final_style_words[:amount_of_vecs_per_class, None, :].to(RAM_Device)*0 + self.content_features.to(RAM_Device)[None, :, :]), 0)
        style_content_features_no = final_style_words_no.to(RAM_Device)[:, None, :] + self.content_features.to(RAM_Device)[None, :, :]

        flattened_sc_features = torch.flatten(style_content_features, end_dim=1)
        flattened_sc_no  = torch.flatten(style_content_features_no, end_dim=1)

        input = torch.cat((flattened_sc_features, flattened_sc_no), 0).to(RAM_Device)
        self.input = input
        targets = torch.tensor(range(len(self.content_features))).repeat(
            1*len(flattened_sc_no)+len(flattened_sc_features), 1).flatten().to(RAM_Device)


        self.lin_model = Linear(self.text_feature_dim, len(self.content_features)).to(self.device)
        lin_optimizer = torch.optim.SGD(self.lin_model.parameters(), lr=0.05, momentum=0.9)


        batch_size_to_remember = 128
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
                loss = 1 * torch.nn.CrossEntropyLoss()(lin_model_pred, batch_target) + 1 * loss_af

                print(loss.cpu().detach().numpy(), end="||")
                loss.backward(retain_graph=True)
                lin_optimizer.step()

                batchsize = batch_size_to_remember
            print("epoch "+ str(n_lin_epoch)+ " done.")
        self.lin_model.eval()

        class_correct = 0
        intermediate_total = 0
        self.clip_model = self.clip_model.to(self.device)
        with torch.no_grad():
            for execution_nr in range(self.dataloader.__len__()):  # complex way to write sample ove rall samples
                features, labels, _,  paths = next(iter(self.dataloader))
                intermediate_total += len(labels)
                class_correct += self.do_test(features, labels, paths)
                print("currently ", class_correct.cpu().detach().numpy(), "ouf of ", intermediate_total, " correct. (", class_correct.cpu().detach().numpy()/(intermediate_total)*100,"%)" )
            print("total accuracy:", 100.0 * class_correct / intermediate_total)
        # return


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
