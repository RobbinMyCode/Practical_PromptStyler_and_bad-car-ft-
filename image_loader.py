import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import torchvision
BICUBIC = InterpolationMode.BICUBIC
#self.clip_model, self.image_preprocess = CLIP.load(self.args.CLIP, device=self.device)


class CheapTestImageDataset(Dataset):
    def __init__(self, base_path, domains, class_names):
        self.domains = domains
        self.class_names = class_names
        if isinstance(domains, str):
            self.img_dirs  = [base_path + "/" + domains + "/" + c_name + "/" for c_name in class_names]
            self.file_names = np.array([np.array([f for f in listdir(dir) if isfile(join(dir, f))])
                                                  for dir in self.img_dirs])
            self.file_lengths_per_class_and_domain = np.array([len(files) for files in self.file_names])
            #print(self.file_lengths_per_class_and_domain)
            self.class_domain_start_index = [[0] for x in self.file_lengths_per_class_and_domain]
            #print(self.class_domain_start_index, len(self.class_domain_start_index), len(self.class_domain_start_index[0]))

            n_classes = len(class_names)
            n_domains = 1
            for c in range(n_classes):

                if c == 0:
                    continue
                else:
                    self.class_domain_start_index[c][0] += (self.class_domain_start_index[c-1][0] +
                                                            self.file_lengths_per_class_and_domain[c-1])



        else:
            self.img_dirs = [[base_path + "/" + domain + "/" + c_name + "/" for domain in domains] for c_name in class_names]

            self.file_names = np.array([np.array([np.array([f for f in listdir(dir) if isfile(join(dir, f))])
                                          for dir in dirs_domain]) for dirs_domain in self.img_dirs])

            self.file_lengths_per_class_and_domain = np.array([np.array([len(files) for files in dirs_domain])
                                                           for dirs_domain in self.file_names])

            self.class_domain_start_index = [[0 for x in dir_domain] for dir_domain in self.file_lengths_per_class_and_domain]
            n_classes = len(class_names)
            n_domains = len(domains)
            for c in range(n_classes):
                for d in range(n_domains):
                    if c==0 and d==0:
                        continue
                    elif d != 0:
                        self.class_domain_start_index[c][d] += (self.class_domain_start_index[c][(d-1)] +
                                                                self.file_lengths_per_class_and_domain[c][(d-1)])
                    elif d == 0:
                        self.class_domain_start_index[c][d] += (self.class_domain_start_index[c-1][n_domains-1] +
                                                                self.file_lengths_per_class_and_domain[c-1][n_domains-1])

        self.n_classes = n_classes
        self.n_domains = n_domains
        self.labels = class_names

    def __len__(self):
        return np.sum(self.file_lengths_per_class_and_domain)

    def __getitem__(self, idx):
        domain_i = 0
        class_n = 0

        for i in range(len(self.class_domain_start_index)):
            if idx >= self.class_domain_start_index[i][0]:
                class_n = i
            else:
                break

        for j in range(len(self.class_domain_start_index[class_n])):
            if idx >= self.class_domain_start_index[class_n][j]:
                #idx -= self.class_domain_start_index[class_n][j]
                domain_i = j

        idx_in_class_and_domain = idx - self.class_domain_start_index[class_n][domain_i]
        if isinstance(self.domains, str):
            img_path = os.path.join(self.img_dirs[class_n], self.file_names[class_n][idx_in_class_and_domain])
        else:
            img_path = os.path.join(self.img_dirs[class_n][domain_i], self.file_names[class_n][domain_i][idx_in_class_and_domain])


        #image = read_image(img_path)
        label = class_n
        domain = self.domains[domain_i]

        return 1, label, domain, img_path

class CheapFasterDataset(Dataset):
    def __init__(self, base_path, domains, class_names):
        self.domains = domains
        self.class_names = class_names
        if isinstance(domains, str):
            self.img_dirs  = [base_path + "/" + domains + "/" + c_name + "/" for c_name in class_names]
            self.file_names = np.array([np.array([f for f in listdir(dir) if isfile(join(dir, f))])
                                                  for dir in self.img_dirs])
            self.file_lengths_per_class_and_domain = np.array([len(files) for files in self.file_names])
            #print(self.file_lengths_per_class_and_domain)
            self.class_domain_start_index = [[0] for x in self.file_lengths_per_class_and_domain]
            #print(self.class_domain_start_index, len(self.class_domain_start_index), len(self.class_domain_start_index[0]))

            n_classes = len(class_names)
            n_domains = 1
            for c in range(n_classes):

                if c == 0:
                    continue
                else:
                    self.class_domain_start_index[c][0] += (self.class_domain_start_index[c-1][0] +
                                                            self.file_lengths_per_class_and_domain[c-1])



        else:
            self.img_dirs = [[base_path + "/" + domain + "/" + c_name + "/" for domain in domains] for c_name in class_names]

            self.file_names = np.array([np.array([np.array([f for f in listdir(dir) if isfile(join(dir, f))])
                                          for dir in dirs_domain]) for dirs_domain in self.img_dirs])

            self.file_lengths_per_class_and_domain = np.array([np.array([len(files) for files in dirs_domain])
                                                           for dirs_domain in self.file_names])

            self.class_domain_start_index = [[0 for x in dir_domain] for dir_domain in self.file_lengths_per_class_and_domain]
            n_classes = len(class_names)
            n_domains = len(domains)
            for c in range(n_classes):
                for d in range(n_domains):
                    if c==0 and d==0:
                        continue
                    elif d != 0:
                        self.class_domain_start_index[c][d] += (self.class_domain_start_index[c][(d-1)] +
                                                                self.file_lengths_per_class_and_domain[c][(d-1)])
                    elif d == 0:
                        self.class_domain_start_index[c][d] += (self.class_domain_start_index[c-1][n_domains-1] +
                                                                self.file_lengths_per_class_and_domain[c-1][n_domains-1])

        self.n_classes = n_classes
        self.n_domains = n_domains
        self.labels = class_names

        for domain_i in range(n_domains):
            for class_n in range(n_classes):
                idx_limit = self.file_lengths_per_class_and_domain[n_classes] if isinstance(self.domains, str) else self.file_lengths_per_class_and_domain[domain_i][n_classes]
                for idx in range(idx_limit):
                    if isinstance(self.domains, str):
                        img_path = os.path.join(self.img_dirs[class_n],
                                            self.file_names[class_n][idx])
                    else:
                        img_path = os.path.join(self.img_dirs[class_n][domain_i], self.file_names[class_n][domain_i][idx])



                #idx_in_class_and_domain = idx - self.class_domain_start_index[class_n][domain_i]



    def __len__(self):
        return np.sum(self.file_lengths_per_class_and_domain)

    def __getitem__(self, idx):
        domain_i = 0
        class_n = 0




        while(idx >= self.class_domain_start_index[class_n][domain_i]):
            if domain_i < self.n_domains-1:
                domain_i += 1
            else:
                class_n += 1
                domain_i = 0
            if class_n == self.n_classes:
                break


        #loop until idx < start_index of class/domain --> -1 as idx must be > start_index (as in m-th sample of nth class/domain)
        if domain_i != 0:
            domain_i -= 1
        else:
            class_n -= 1
            domain_i = self.n_domains-1

        idx_in_class_and_domain = idx - self.class_domain_start_index[class_n][domain_i]
        #print("idx", idx, "domain:", domain_i, "/", self.n_domains, "class:", class_n, "/", self.n_classes,
        #      "remainder", idx_in_class_and_domain, "images in dir", self.file_lengths_per_class_and_domain[class_n][domain_i])
        #if idx_in_class_and_domain == 740:
        #    print(self.class_domain_start_index)
        #    print(self.file_lengths_per_class_and_domain)

        if isinstance(self.domains, str):
            img_path = os.path.join(self.img_dirs[class_n], self.file_names[class_n][idx_in_class_and_domain])
        else:
            img_path = os.path.join(self.img_dirs[class_n][domain_i], self.file_names[class_n][domain_i][idx_in_class_and_domain])

        #image = read_image(img_path)
        label = class_n
        domain = self.domains[domain_i]

        return 1, label, domain, img_path
class CustomImageDataset(Dataset):
    def __init__(self, base_path, class_name, transform=None, target_transform=None):
        self.img_labels = class_name #pd.read_csv(annotations_file)
        self.img_dir = base_path +"/" + class_name
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



class CheapTestImageDataset_image_outs(Dataset):
    def __init__(self, base_path, domains, class_names, args=None, further_transforms=[]):
        if args!=None:
            device = torch.device("cuda:" + args.GPU_num if torch.cuda.is_available() else "cpu")
            model, self.image_preprocess = clip.load(args.CLIP, device=device)
            n_px = 224 #model.input_resolution.item()
            if len(further_transforms) == 0:
                self.clip_transform = Compose([Resize(n_px, interpolation=BICUBIC),
                    CenterCrop(n_px),
                    #ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), ])

        self.domains = domains
        self.class_names = class_names
        if isinstance(domains, str):
            self.img_dirs  = [base_path + "/" + domains + "/" + c_name + "/" for c_name in class_names]
            self.file_names = np.array([np.array([f for f in listdir(dir) if isfile(join(dir, f))])
                                                  for dir in self.img_dirs])
            self.file_lengths_per_class_and_domain = np.array([len(files) for files in self.file_names])
            #print(self.file_lengths_per_class_and_domain)
            self.class_domain_start_index = [[0] for x in self.file_lengths_per_class_and_domain]
            #print(self.class_domain_start_index, len(self.class_domain_start_index), len(self.class_domain_start_index[0]))

            n_classes = len(class_names)
            n_domains = 1
            for c in range(n_classes):

                if c == 0:
                    continue
                else:
                    self.class_domain_start_index[c][0] += (self.class_domain_start_index[c-1][0] +
                                                            self.file_lengths_per_class_and_domain[c-1])



        else:
            self.img_dirs = [[base_path + "/" + domain + "/" + c_name + "/" for domain in domains] for c_name in class_names]

            self.file_names = np.array([np.array([np.array([f for f in listdir(dir) if isfile(join(dir, f))])
                                          for dir in dirs_domain]) for dirs_domain in self.img_dirs])

            self.file_lengths_per_class_and_domain = np.array([np.array([len(files) for files in dirs_domain])
                                                           for dirs_domain in self.file_names])

            self.class_domain_start_index = [[0 for x in dir_domain] for dir_domain in self.file_lengths_per_class_and_domain]
            n_classes = len(class_names)
            n_domains = len(domains)
            for c in range(n_classes):
                for d in range(n_domains):
                    if c==0 and d==0:
                        continue
                    elif d != 0:
                        self.class_domain_start_index[c][d] += (self.class_domain_start_index[c][(d-1)] +
                                                                self.file_lengths_per_class_and_domain[c][(d-1)])
                    elif d == 0:
                        self.class_domain_start_index[c][d] += (self.class_domain_start_index[c-1][n_domains-1] +
                                                                self.file_lengths_per_class_and_domain[c-1][n_domains-1])

        self.n_classes = n_classes
        self.n_domains = n_domains
        self.labels = class_names

    def __len__(self):
        return np.sum(self.file_lengths_per_class_and_domain)

    def __getitem__(self, idx):
        domain_i = 0
        class_n = 0

        for i in range(len(self.class_domain_start_index)):
            if idx >= self.class_domain_start_index[i][0]:
                class_n = i
            else:
                break

        for j in range(len(self.class_domain_start_index[class_n])):
            if idx >= self.class_domain_start_index[class_n][j]:
                # idx -= self.class_domain_start_index[class_n][j]
                domain_i = j

        idx_in_class_and_domain = idx - self.class_domain_start_index[class_n][domain_i]
        if isinstance(self.domains, str):
            img_path = os.path.join(self.img_dirs[class_n], self.file_names[class_n][idx_in_class_and_domain])
        else:
            img_path = os.path.join(self.img_dirs[class_n][domain_i],
                                    self.file_names[class_n][domain_i][idx_in_class_and_domain])
        #image = read_image(img_path)
        label = class_n
        domain = self.domains[domain_i]
        image = torchvision.io.read_image(img_path)

        return self.clip_transform(image), label
