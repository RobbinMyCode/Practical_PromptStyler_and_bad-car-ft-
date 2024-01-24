import torch
import numpy

implementations = ["ResNet-50", "ViT-B/16", "ViT-L/14"]
dim_style_vector = {
    "ResNet-50": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768
}
D = dim_style_vector

dim_text_image_feature = {
    "ResNet-50": 1024,
    "ViT-B/16": 512,
    "ViT-L/14": 768
}
C = dim_text_image_feature
n_training_iter = 100
lr = 0.002
momentum = 0.9

n_style_vectors = 80
K = n_style_vectors

#dummyvariable_so its usable
used_net = implementations[0]

style_word_vectors = torch.normal(0, 0.02, size=(n_style_vectors, D[used_net]))

def L_style(encoded_features):
    i =  len(encoded_features)
    L = 0
    for j in range(i-1):
        L += 1/(i-1) * torch.abs(encoded_features[i] / torch.linalg.norm(encoded_features[i])
                                  @ (encoded_features[j] / torch.linalg.norm(encoded_features[j]))
                                  )
    return L



for i in range(K):
    for iter in range(n_training_iter):
        continue;

