from tracemalloc import start
from transformers import BertTokenizer
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn


def randomMask(now_text, now_mask, tokenizer):
    word_tokens = tokenizer.tokenize('[SEP]' + now_text + '[CLS]')

    sum_class = 0
    position = list()
    for k,token in enumerate(now_mask):
        if token == 2:
            position.append(k)
            sum_class = sum_class + 1
    random_number = int(sum_class * torch.rand(1).item())
    start_index = position[random_number]
    for j in range(start_index,len(now_mask)):
        if now_mask[j] == 0:
            break
        else:
            word_tokens[j] = '[MASK]'
            end_index = j
    

    mask_text = tokenizer.convert_tokens_to_string(word_tokens)[6:-6]

    mask_word_tokens = tokenizer.tokenize('[SEP]' + now_text + '[CLS]')[start_index : end_index + 1]
    mask_word = tokenizer.convert_tokens_to_string(mask_word_tokens).replace(" ","")

    return mask_text, mask_word

def cosine_similarity(x, y, norm=False):

    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = np.zeros(len(x))

    if any(x == zero_list) or any(y == zero_list):
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return cos

def compute_sim(now_image, now_text, tmp_imgs, tmp_texts, beta = 0.3):

    sim_list = list()
    for i,tmp_img in enumerate(tmp_imgs):
        tmp_text = tmp_texts[i]
        sim = beta * cosine_similarity(now_text[0], tmp_text[0]) + (1 - beta) * cosine_similarity(now_image[0], tmp_img[0])
        sim_list.append(sim)
    temp=[]
    Inf = 0
    if len(sim_list) == 0:
        return 0
    for i in range(70):
        temp.append(sim_list.index(max(sim_list)))
        sim_list[sim_list.index(max(sim_list))]=Inf
    temp.sort()
    return temp

def find_negative_sample(index_find, beta, mask_word, id2text, attr_dict, mask_attr, 
                        text_embedding, I, resnet_embedding, image2embedding, now_image_embedding, nowtext_embedding, mask_negative_texts, 
                        negative_224s = None, negative_img_features = None):

    tmp_texts_embeddings = list()
    tmp_texts = list()
    tmp_index = list()
    tmp_imgs = list()
    tmp_224s = list()
    for i in range(1, 500, 1):
        if mask_word not in id2text[str(I[0][i])]:
            if i not in index_find:
                mask_part_text = id2text[str(I[0][i])].split('|')[attr_dict[mask_attr]]
                if mask_part_text.replace(" ","").replace(".","").split(":")[1] != "" :
                    tmp_texts_embeddings.append(text_embedding[I[0][i]:I[0][i]+1])
                    tmp_imgs.append(resnet_embedding[I[0][i]])
                    tmp_224s.append(image2embedding[I[0][i]].cuda())
                    tmp_texts.append(id2text[str(I[0][i])])
                    tmp_index.append(i)

    top2_similarity_index = compute_sim(now_image_embedding, nowtext_embedding, tmp_imgs, tmp_texts_embeddings, beta)
    if top2_similarity_index == 0:
        return 0, 0, 0, 0
    for i in top2_similarity_index:
        index_find.append(tmp_index[i])
        mask_negative_texts.append(tmp_texts[i])
        if len(index_find) == 1:
            negative_224s = tmp_224s[i]
            negative_img_features = tmp_imgs[i]
        else:
            negative_224s = torch.cat((negative_224s, tmp_224s[i]), 0)
            negative_img_features = np.concatenate((negative_img_features, tmp_imgs[i]), axis = 0)
    
    return index_find, negative_224s, negative_img_features, mask_negative_texts

def compute_construct_loss(now_image, ori_mask_text,
                positive_img_feature, mask_positive_text,
                negative_img_features, mask_negative_texts,
                model, w2v_seen):

    image = torch.cat((now_image.unsqueeze(0), positive_img_feature, negative_img_features), 0)
    texts = [ori_mask_text] + [mask_positive_text] + mask_negative_texts

    embedding = model(image,w2v_seen,texts,contrast=True)
    original_embedding = embedding[0:1]
    positive_embedding = embedding[1:2]
    negative_embedding = embedding[2:]

    # positive logits: Nx1
    l_pos = torch.einsum('nc,nc->n', [original_embedding, positive_embedding]).unsqueeze(-1)
	# negative logits: NxK
    l_neg = torch.einsum('nc,ck->nk', [original_embedding, negative_embedding.t()])
	
	# logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)
	# labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    criterion = nn.CrossEntropyLoss()
	# compute output
    loss = criterion(logits, labels)
    return loss


import os
import json
def matrix2text(matrix, data, name2classindex,opt,mapping):
    min_max_scaler = preprocessing.MinMaxScaler() 
    w2v_01 = min_max_scaler.fit_transform(data.w2v.reshape(-1, 1)).reshape(data.w2v.shape[0],-1)

    text = ""
    for i, key in enumerate(name2classindex):
        value = name2classindex[key]
        prompt = key + " : "
        for j,va in enumerate(value):
            if matrix[va].item() == 1:
                if opt.dataset == "SUN":
                    if str(va) in mapping:
                        prompt = prompt + " " + mapping[str(va)] + ","
                    else:
                        prompt = prompt + " " + data.seg_class[va].strip() + ","
                elif opt.dataset == "AWA2":
                        prompt = prompt + data.seg_class[va].strip() + ","
                else:
                    if str(va) in mapping:
                        prompt = prompt + str(va) + mapping[str(va)] + ","
                    else:
                        prompt = prompt + str(va) + data.seg_class[va].split("::")[1].replace("-"," ").replace("_"," ").split("(")[0].strip() + ","
        if prompt[-1] == ',':
            prompt = prompt[:-1]
        if i == len(name2classindex)-1:
            text = text + prompt + '.'
        else:
            text = text + prompt + " | "
    return text

def replace_maskword(opt, now_text, mask_word):


    if opt.dataset == "SUN":
        if mask_word == "enclosed":
            now_text = now_text.replace('semi enclosed', 'helloooo1')
            now_text = now_text.replace(mask_word, '[MASK]')
            now_text = now_text.replace('helloooo1', 'semi enclosed')
        elif mask_word == "bathing":
            now_text = now_text.replace('sunbathing', 'helloooo1')
            now_text = now_text.replace(mask_word, '[MASK]')
            now_text = now_text.replace('helloooo1', 'sunbathing')
        else:
            now_text = now_text.replace(mask_word,'[MASK]')
    elif opt.dataset == "AWA2":
        if mask_word == 'meat':
            now_text = now_text.replace('meatteeth', 'helloooo1')
            now_text = now_text.replace(mask_word, '[MASK]')
            now_text = now_text.replace('helloooo1', 'meatteeth')
        elif mask_word == 'cave':
            now_text = now_text.replace('scavenger', 'helloooo1')
            now_text = now_text.replace(mask_word, '[MASK]')
            now_text = now_text.replace('helloooo1', 'scavenger')
        elif mask_word == 'active':
            now_text = now_text.replace('inactive', 'helloooo1')
            now_text = now_text.replace(mask_word, '[MASK]')
            now_text = now_text.replace('helloooo1', 'inactive')
        else:
            now_text = now_text.replace(mask_word,'[MASK]')
    else:
        now_text = now_text.replace(mask_word,'[MASK]')
    return now_text

import copy
from sklearn import preprocessing
import random
import numpy as np
import re
import math
softmax = nn.Softmax(dim=0)
min_max_scaler = preprocessing.MinMaxScaler() 


def prepare_original_sample(matrix, index, opt, data, classindex2name, name2classindex, classid2length, tokenizer, mapping):

    matrix_now = copy.deepcopy(matrix[index])

    matrix_index = index

    cannotmask_index = list()
    w2v_notmiss_index = list()
    w2v_miss_index = list()


    random_number = random.randint(1, len(matrix_index)) - 1        
    mask_index = matrix_index[random_number]
    if opt.dataset == "RSSDIVCS":
            if str(mask_index) in mapping:
                mask_w2v = mapping[str(mask_index)]

    
    for j in range(len(matrix_now)):
        if matrix_now[j] == 1 and (torch.rand(1).item()> 1 - opt.w2v_miss) and j!= mask_index:
            matrix_now[j] = 0
            if j in cannotmask_index:
                w2v_miss_index.append(j)
        else:
            if j in cannotmask_index and j != mask_index:
                w2v_notmiss_index.append(j)

    now_text = matrix2text(matrix_now, data, name2classindex, opt,mapping)

    return now_text, mask_w2v, mask_index, matrix_index