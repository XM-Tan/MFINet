import os
import sys
import copy
import json
import random

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from sklearn import preprocessing
from transformers import BertTokenizer

from log import Log
from opt import get_opt
import visual_utils
from visual_utils import prepare_semantic_label
from model_proto import Multi_attention_Model
from main_utils import test_zsl, test_gzsl, get_loader, Loss_fn, Result
import datetime

import pandas as pd

cudnn.benchmark = True
opt = get_opt()

zsl = "zsl"

curr_time = datetime.datetime.now()
log_name = str(curr_time) + "-" + zsl

logger = Log(os.path.join('./log',opt.dataset), log_name).get_logger()
logger.info(json.dumps(vars(opt)))

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)


def main(): 
    data = visual_utils_my_all.DATA_LOADER(opt)

    opt.data = data

    class_semantic = data.seg_features
    label_semantic = data.seg_class               
    label_semantic = label_semantic.reshape(-1, 1)     

    opt.semantic_feature = data.seg_features
    semantic_seen_feature = prepare_semantic_label(opt.semantic_feature, data.seenclasses).t()  
    data.semantic_seen_feature = semantic_seen_feature     

    semantic_zsl = prepare_semantic_label(class_semantic, data.unseenclasses).cuda() #(300,10)
    semantic_seen = prepare_semantic_label(class_semantic, data.seenclasses).cuda()  #(300,60)
    semantic_gzsl = torch.transpose(class_semantic, 1, 0).cuda()  #(300,70)
    semantic_deal = copy.deepcopy(prepare_semantic_label(class_semantic, data.seenclasses)).t()
    min_max_scaler = preprocessing.MinMaxScaler() 
    semantic_deal = min_max_scaler.fit_transform(semantic_deal.reshape(-1, 1)).reshape(semantic_deal.shape[0],-1)

    semantic_seen_label = []
    for i in data.seenclasses:
        semantic_seen_label.append(label_semantic[i][0][0])
    semantic_zsl_label=[]
    for i in data.unseenclasses:
        semantic_zsl_label.append(label_semantic[i][0][0])

    ManhattanDistance = torch.tensor(np.ones([semantic_deal.shape[0],semantic_deal.shape[0]]))
    for i in range(semantic_deal.shape[0]):
        for j in range(semantic_deal.shape[0]):
            if i != j:
                ManhattanDistance[i][j] = np.sum(np.fabs(semantic_deal[i] - semantic_deal[j]))
    ManhattanDistance=torch.pow(ManhattanDistance, 2)

    trainloader, testloader_unseen, visloader = get_loader(opt, data)
   

    logger.info('Create Model...')
    model_baseline = Multi_attention_Model(opt, using_amp=True)

    criterion = nn.CrossEntropyLoss()

    criterion_regre = nn.MSELoss()
    if opt.class_embedding == "w2v":
        semantic = 'w2v'
    elif opt.class_embedding == "bert":
        semantic = 'bert'
    reg_weight = {'final': {'xe': opt.xe, semantic: opt.semantic, 'regular': opt.regular}}  # 权重
    if torch.cuda.is_available():
        model_baseline = model_baseline.cuda()
        semantic_seen = semantic_seen.cuda()
        semantic_zsl = semantic_zsl.cuda()
        semantic_gzsl = semantic_gzsl.cuda()

    result_zsl = Result()
    result_gzsl = Result()


    with open(os.path.join("/work/tanxm/duet2my/cache/", opt.dataset, "classindex2name.json"),"r") as f:
        classindex2name = json.load(f)
    name2classindex = dict()

    for key,value in classindex2name.items():
        if value not in name2classindex:
            name2classindex[value] = [int(key)]
        else:
            name2classindex[value].append(int(key))
    
    ori_seenclass2imageindexs = dict()
    for seenclass in data.seenclasses:
        ori_seenclass2imageindexs[seenclass.item()] = np.where(data.label == seenclass.item())[0].tolist()
    seenclass2imageindexs = copy.deepcopy(ori_seenclass2imageindexs)


    if opt.only_evaluate:
        logger.info('Evaluate ...')
        model_baseline.load_state_dict(torch.load(opt.resume))
        model_baseline.eval()

        acc_ZSL = test_zsl(opt, model_baseline, testloader_unseen, data.unseenclasses)
        logger.info('ZSL test accuracy is {:.1f}%'.format(acc_ZSL))
    else:
        logger.info('Train and test...')
        for epoch in range(opt.nepoch):  
            model_baseline.train()
            current_lr = opt.classifier_lr * (0.8 ** (epoch // 10))   
            params_for_optimization = model_baseline.parameters()
            optimizer = optim.Adam([p for p in params_for_optimization if p.requires_grad], lr=current_lr)
            loss_log = {'ave_loss': 0, 'l_xe_final': 0, 'l_semantic_final': 0, 'l_regular_final': 0,
                        'l_xe_layer': 0, 'l_semantic_layer': 0, 'l_regular_layer': 0, 'l_cpt': 0}
            batch = len(trainloader) 


            for i_realindex, (batch_input, batch_target, impath, matrix, img_loc) in tqdm(enumerate(trainloader), total = len(trainloader)):
                class_target = batch_target
                batch_target = visual_utils_my_all.map_label(batch_target, data.seenclasses)
                input_v = Variable(batch_input)
                label_v = Variable(batch_target)

                if opt.cuda:
                    input_v = input_v.cuda()
                    label_v = label_v.cuda()
                visual_label = []
                for index_impath in impath:
                    visual_label_index = int(str(index_impath).split("/")[5]) - 1
                    visual_label1 = classindex2name[str(visual_label_index)] 
                    visual_label.append(visual_label1)
 
                output, pre_semantic, _, _, mask_loss, image_embedding_class, embedding_for_sc = model_baseline(input_v, semantic_seen, visual_label, is_mask=False, whole_semantic=semantic_gzsl)
  
                label_a = semantic_seen[:, label_v].t()
                loss = (Loss_fn(opt, loss_log, reg_weight, criterion, criterion_regre, model_baseline, output, pre_semantic, label_a, label_v, embedding_for_sc) + mask_loss * opt.mask_loss_xishu) / opt.gradient_time
                loss_log['ave_loss'] += loss.item()

                loss.backward()
                if (i_realindex+1) % opt.gradient_time == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    key=0
                else:
                    key=1

                if (i_realindex + 1) != batch and key == 1:
                    continue

                if ((opt.dataset == "RSSDIVCS") and (((i_realindex + 1) == batch) or((epoch>=5) and (i_realindex +1) == batch/2/opt.gradient_time*opt.gradient_time))):
                    logger.info('\n[Epoch %d, Batch %5d] Train loss: %.3f '% (epoch+1, batch, loss_log['ave_loss'] / batch))
                    model_baseline.eval()

                    acc_ZSL, testpredict_class = test_zsl(opt, model_baseline, testloader_unseen, semantic_zsl, data.unseenclasses, classindex2name)

                    if acc_ZSL > result_zsl.best_acc:
                        patient = 0
                    else:
                        patient = patient + 1
                        print("Counter {} of {}".format(patient,opt.patient))
                        logger.info("Counter {} of {}".format(patient,opt.patient))
                        if patient > opt.patient:
                            print("Early stopping with best_acc: ", result_zsl.best_acc, "and val_acc for this epoch: ", acc_ZSL, "...")
                            logger.info('Early stopping with best_acc: %s and val_acc for this epoch: %s ...' % (result_zsl.best_acc,acc_ZSL))
                            sys.exit()
                    result_zsl.update(epoch+1, acc_ZSL, step = 0.0)
                    logger.info('\n[Epoch {}] ZSL test accuracy is {:.1f}%, Best_acc [{:.1f}% | Epoch-{}]'.format(epoch+1, acc_ZSL, result_zsl.best_acc, result_zsl.best_iter))
  

if __name__ == '__main__':
    main()
