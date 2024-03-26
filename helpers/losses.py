import os
import argparse
import torch
import clip
from torch import nn
from torch.nn import functional as F
from datetime import datetime
from helpers.image_loader import CheapTestImageDataset
import helpers.pseudoCLIP as pseudoCLIP
import numpy as np
import pickle

from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True


def style_loss(style_vec, style_index=1):
    if style_index == 0:
        return torch.tensor(0)
    else:
        loss = 0

        norm_i = torch.linalg.norm(style_vec[style_index])
        for j in range(0, style_index):
            #loss += torch.abs(nn.CosineSimilarity)
            loss += torch.abs(1./style_index * style_vec[style_index] @ style_vec[j] /
                     (norm_i * torch.linalg.norm(style_vec[j])))
        return loss

def content_loss(style_content_words, content_words, style_index=1, n_classes=7):
    '''
        loss to maximize cosine similarity with style_content_vector and corresponding content_vector while minimizing all other similarities
        - exp for softmax and log for loss scaling [0,1]-> [infty, 0]
    '''
    if style_content_words.dtype !=  content_words.dtype:
        style_content_words = style_content_words.to(content_words.dtype)
    #print("used style_content_word", style_content_words[style_index])
    z = style_content_words[style_index] @ content_words.T / (
            torch.linalg.norm(style_content_words[style_index], axis=-1) * torch.linalg.norm(content_words, axis=-1))
    #print(z.size())
    #exit()
    z = torch.exp(z)
    sum_z_imn = torch.sum(z, dim=-1)

    z_diag = torch.diagonal(z, 0)


    loss = -1. / n_classes* torch.sum( torch.log(z_diag) - torch.log(sum_z_imn) )
    #print("content loss = ", loss)
    return loss

def style_content_loss(model_output, content_labels, style_loss_factor=1, loss_goal_values = [0,0], style_index=0, n_classes=7, verbose=1):
    '''
        PromptStyler style-content-loss
    :param model_output: predictions
    :param content_labels: labels
    :param style_loss_factor: loss = style_loss_factor * rel_style_loss + rel_content_loss
    :param loss_goal_values: rel_style_loss = abs( style_loss  - loss_goal_values[0]), and content analogeous
    :param style_index: at which index the comparison is (only with indexes before that)
    :param n_classes: num_classes: requires for regularization of content loss
    :param verbose: if 1: prints loss, style loss and content loss
    :return: corresponding loss
    '''
    loss_content = content_loss(model_output[1], content_labels, style_index, n_classes)
    loss_style = style_loss(model_output[0], style_index)

    loss = torch.abs(loss_content-loss_goal_values[0]) + style_loss_factor * torch.abs(loss_style - loss_goal_values[1])
    if verbose == 1:
        print("", end="\r")
        print("loss: ", loss.detach().cpu().numpy(), "\t style loss:", loss_style.detach().cpu().numpy(), "content loss", loss_content.detach().cpu().numpy(), end="\t")
    if torch.isnan(loss):
        print(loss)
        exit("nan loss occured")
    return loss

def arcface_loss(linear_output, linear_input, linear_weights, labels, s=5, m=0.5):
    #arcface_softmax
    cos_m = torch.cos(torch.tensor(m))
    sin_m = torch.sin(torch.tensor(m))

    norm_weights = torch.sqrt(torch.sum(linear_weights ** 2, axis=-1)).unsqueeze(0)
    norm_input = torch.sqrt(torch.sum(linear_input ** 2, axis=-1)).unsqueeze(-1)

    # from out = in @ weights = |in| * |weights| *cos(theta) --> cos(theta) = out/(|in| * |weights|)
    cos_theta = linear_output / (norm_input * norm_weights)

    sin_theta = (1 - cos_theta ** 2) ** 0.5
    cos_theta_plus_m = cos_theta * cos_m - sin_theta * sin_m  # expansion formula of cos(a+b)

    exp_pos = torch.sum(torch.exp(s * cos_theta_plus_m) * F.one_hot(labels, num_classes=-1), axis=-1)
    exp_neg = torch.exp(s * cos_theta)

    #denominator_sum_all = torch.sum(exp_neg, axis=-1)
    denominator_sum = torch.sum(exp_neg * (1 - F.one_hot(labels, num_classes=-1)), axis=-1)

    target = exp_pos / (denominator_sum + exp_pos)
    error = - torch.log(target)
    return torch.mean(error)
def arcface_softmax(linear_output, linear_input, linear_weights, s=5, m=0.5):
    '''
        computes the arcface softmax of a linear layer (which must not use a bias)
    :param self:
    :param linear_output: output of linear layer (of which to compute the arcface softmax)
    :param linear_input: input to linear layer
    :param linear_weights: linear.weight (for linear layer)
    :param s: scaling hyperparameter of arcface
    :param m: angle parameter of arcface
    :return:
    '''
    cos_m = torch.cos(torch.tensor(m))
    sin_m = torch.sin(torch.tensor(m))

    norm_weights = torch.sqrt(torch.sum(linear_weights ** 2, axis=-1)).unsqueeze(0)
    norm_input = torch.sqrt(torch.sum(linear_input ** 2, axis=-1)).unsqueeze(-1)

    #from out = in @ weights = |in| * |weights| *cos(theta) --> cos(theta) = out/(|in| * |weights|)
    cos_theta = linear_output / (norm_input * norm_weights)

    sin_theta = (1 - cos_theta ** 2) ** 0.5
    cos_theta_plus_m = cos_theta * cos_m - sin_theta * sin_m  #expansion formula of cos(a+b)

    exp_pos = torch.exp(s * cos_theta_plus_m)
    exp_neg = torch.exp(s * cos_theta)

    denominator_sum_all = torch.sum(exp_neg, axis=-1)
    result = exp_pos

    for i in range(result.size(1)):
        result[:, i] /= (denominator_sum_all - exp_neg[:, i] + exp_pos[:, i]) # / all neg but i
    #for i in range(15):  # resultb.size(0)):
    #print(result[:5])
    return result
