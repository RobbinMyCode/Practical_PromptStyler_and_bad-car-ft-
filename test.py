import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
#from pkg_resources import packaging

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import clip
import pseudoCLIP
import numpy as np
import sys
import pickle
clip_model, image_preprocess = pseudoCLIP.load("ViT-B/16", device="cuda")
#token = torch.cat([clip.tokenize(f"{c}") for c in ["a photo of a cat", "a sketch of a cat"]]).to("cuda")
text = "a b of a b"#["a art of a", "cat", "d0g"]
#style_words = torch.load('values.pt')
token = clip.tokenize(text).to("cuda")
pseudowords = torch.randn((80, 512)) * 0.02
encoding = clip_model.encode_text(token, pseudowords, position_pseudo=2, print_intermediate=False).to(torch.float32).to("cuda")
print(encoding.size())
print(sys.getsizeof(encoding), sys.getsizeof(torch.tensor(1, dtype=torch.float16)))
file = open("newfile.txt", "w")
np.savetxt(file, encoding.detach().cpu().numpy())

with open('newfile.pickle', 'wb') as fp:
    pickle.dump(encoding.detach().cpu().numpy(), fp)

test = torch.ones((4,3,10))
for i in range(4):
    for j in range(3):
        for k in range(10):
            test[i,j,k] = 10*i + j + 0.1*k

print(test, test.size())
print(torch.flatten(test, end_dim=-2), torch.flatten(test, end_dim=-2).size())

a = [1,2,3,4,5]
print(a[:1], a[1:])

print("image_encoder weights:")
for param in clip_model.encode_image.parameters():
    print(param)

#for i in range (0,5):
#    matrix = encoding[i]
#    np.savetxt(file, matrix)
#file.close()
#print(encoding)
#print(clipmodel)
#aa = CLIP_pseudo()
#for name, module in clip_model.named_modules():
#    print(name, type(module))
#print(clip_model)

#only first kind: 0.8968
#both variants 0.9392
#only first long training 9355
#both long 0.9387
#only second long 0.9420

#short only classnames 0.951 // 0.9193
#long only classnames 0.9275

#random untrained second + classnames 0.9206
#second with classnames(not weighted) 0.9410
#second with casnames[x80) 0.9420

#a photo of classname 0.9455,
#a photo of a cn. 0.9472, 0.9442 with short training(25)
#second with a photo of classname #0.9469

#L1=a photo of a cn. , cn 0.9468,
#L2=above + words from rise: 0.9577
#L3 = L2 +drawing/sketch/image of a cn. 0.9549
#L4= L3 +painint/cartoon (of a) cn.  0.9579, 0.9523 (early weak strong later)
#L5=l4 + second 0.9574, 0.9573 (strong early later weaker) --> leavin s1,s2,s3 out did not help
#L5_1 = l5 no bias 0.9610 (scnd try 0.9610)
#L5_2 = L5_1 with arcface as softmax 0.9614
#--> with only 10 of second kind 0.9587 (0.9610 at longer training (200 epoch)
#--> 20 0.9623 (200 epochs)   --> 40 0.9547 (200 epochs)
#--> 15 0.9637  (0.9637) --> 17 0.9606  --> 13 0.9572 --> 16 0.9608

#L6 = --> 14 0.9717 (0.9717, 0.9717, 0.9717) normal softmax also 0.9717 = 96.006%

#L6 with VIT/L 0.9871
# --> added style words for a photo of 0.9859 --> remove again
# --> style word split (7/7) as above: 0.9871 (noise was still just on ...class...) ==> they just get trained to be small but orthogonal independend of trained word

#L7: lr=0.05, epochs=15, arcface addition, else L6
# --> added af = 0.002 * arcface_loss 0.9889 [actually 97.705696% as faulty calcs]  EXACT same with 0*arcface_loss
#--> af = 0.1 97.844% [0,9903]
#--> af = 0.5 98.0123% [0.9920]

#==> af = 1 98.071598% [0.9926]
#   --> crossentropy factor cf = 0.1 98.0617% [0.9925]
#   --> cf = 0: 98.0617% [0.9925] => WITH arcfacesoftmax as decider 97.99248% [0.9918]

#L8: L7 with lr vary [af=cf=1]
#--> lr=0.2: 97.8837%
#--> lr=0.1: 98.02215% [0.9921] [scnd 98.02215%]
#--> lr=0.04: 98.0518%
#--> lr=0.06 98.071598% ===>> 0.5 seems fine   VLCS: 76.42299%;  (w.o noise training 75.5766%), ViT/B 74.321%

