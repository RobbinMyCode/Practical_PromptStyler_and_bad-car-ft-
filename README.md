# Practical Domain Generalization: PromptStyler (+ Variants) and (not properly working) CAR-FT

This is an unofficial Implementation of the said methods, 
PromptStyler https://promptstyler.github.io/
CAR-FT: https://arxiv.org/abs/2211.16175.

CAR-FT did not work remotely as good as it should be (and the implementation is not quite perfect),
PromptStyler has slight differences in performance, but works in general
## Getting Started

### Data Preparation
* Download PACS dataset from [here](https://drive.google.com/file/d/1PadzfWayyfyb9idS9n8mP_PjgwfCgLrD/view?usp=sharing)
* Download VLCS dataset from [here](https://drive.google.com/file/d/1VqN_krgoc1qKkO9m__0tCmceuZcDzskc/view?usp=drive_link)
* Download OfficeHome dataset from [here](https://drive.google.com/file/d/1llt8XIdCoYYcf8znposggDRjKtJh1O8X/view?usp=drive_link)
* Download Terra dataset from [here](https://drive.google.com/file/d/1i0O4e7YkW4hUP-nA56LhMSkIpr6rCi1j/view?usp=drive_link)
you might want to rename "Terra_incognita" to "Terra" if its saved with that name


The dataset is structured as follows:
```
dataset
├── PACS
│   ├── Domain1
│   ├── Domain2
│   └── Domain3
│   └── Domain4
├── VLCS
│   ├── ...
├── OfficeHome
│   ├── ...
└── Terra
    ├── ...
```
Terra may be named terra_incognita, if so,  a rename is required.

### Install
* Pytorch 1.7.1 (or later) from [here](https://pytorch.org/)
* CLIP from [here](https://github.com/openai/CLIP) // pip install openai-clip
* Open_clip: pip install open_clip

### Launch a training: vanilla PromptStyler
```
python train_promptstyler.py\
       --dataset "PACS" --seed 0 --output_folder "sweep1" --data_path "your datasets path"  --CLIP "ViT-L/14"
       --norm True --style_word_basis "a some style of a" --style_word_index 1
```

### Launch a training: multiple basis PromptStyler (only useful for Terra)
```
python PS_mult_bas.py\
       --dataset "Terra" --seed 0 --output_folder "sweep1" --data_path "your datasets path"  --CLIP "ViT-L/14"
       --norm False
```

### Launch a training: FULL pseudo words PromptStyler (tends to be not better than the OG)
```
python PS_full_pseudo.py\
       --dataset "Terra" --seed 0 --output_folder "sweep1" --data_path "your datasets path"  --CLIP "ViT-L/14"
       --norm False --number_style_words 15 --pseudo_lengths [4,3] --class_words_index [4,3]
```
### Launch a training/evaluation: Already trained PromptStyler Linear Layer in Split-Image (only specified for Terra, larger improvement)
```
python Linear_split_image.py\
       --dataset "Terra" --seed 0  --data_path "your datasets path"  --CLIP "ViT-L/14"
       --norm False --batch_size 35
```
* requires a PS training done with --save_lin_weights True, to use this classifier
* norm must be the same as in said PS training 
* adapt batch_size to your gpu size (e.g. ~35 for ViT-L @ 8GB)
* For Terra: increases performance by ~8%


### Launch a training: vanilla CAR-FT
```
python train_promptstyler.py\
       --dataset "PACS" --seed 0 --output_folder "sweep1" --data_path "your datasets path" 
       --use_ImageNet_style_words False
```
* ImageNet StyleWords are the default for Terra, if False: use  Prompts Saved by a PromptStyler variant if possible, else "a photo of a {class}"


```

## Acknowledgments
Codebase is originally built upon RISE  ["A Sentence Speaks a Thousand Images: Domain Generalization through Distilling CLIP with Language Guidance"](https://browse.arxiv.org/pdf/2309.12530v1.pdf), which is build opon[OoD-Bench](https://github.com/ynysjtu/ood_bench), [JigenDG](https://github.com/fmcarlucci/JigenDG) and [DomainBed](https://github.com/facebookresearch/DomainBed).
This is mainly visible in the structure of the execution files.

