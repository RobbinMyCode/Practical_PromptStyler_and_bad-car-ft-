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



class CheapTestImageDataset(Dataset):
    def __init__(self, base_path, domains, class_names):
        '''
            inits dataloader: path: base_path/domain/class_name for domain, class_name in zip(domains class_names)
        :param base_path: where dirs for "domains" are in
        :param domains: contains dirs with "class_names" each
        :param class_names: dir_names where the images are
        '''
        self.domains = domains
        self.class_names = class_names
        #-- single domain
        if isinstance(domains, str):
            self.img_dirs  = [base_path + "/" + domains + "/" + c_name + "/" for c_name in class_names]
            self.file_names = [[f for f in listdir(dir) if isfile(join(dir, f))]
                                                  for dir in self.img_dirs]
            self.file_lengths_per_class_and_domain = np.array([len(files) for files in self.file_names])
            self.class_domain_start_index = [[0] for x in self.file_lengths_per_class_and_domain]

            n_classes = len(class_names)
            n_domains = 1

            for c in range(n_classes):
                if c == 0:
                    continue
                else:
                    self.class_domain_start_index[c][0] += (self.class_domain_start_index[c-1][0] +
                                                            self.file_lengths_per_class_and_domain[c-1])


        #-- multiple domains
        else:
            self.img_dirs = [[base_path + "/" + domain + "/" + c_name + "/" for domain in domains] for c_name in class_names]

            self.file_names = [[[f for f in listdir(dir) if isfile(join(dir, f))]
                                          for dir in dirs_domain] for dirs_domain in self.img_dirs]

            self.file_lengths_per_class_and_domain = [[len(files) for files in dirs_domain]
                                                           for dirs_domain in self.file_names]

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
        '''

        :param idx: idx of element to retrieve (0 <= idx < __len__)
        :return: [1, label_idx, domain_idx, img_path(string)]
        '''
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

