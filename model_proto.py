from json import encoder
import torch.nn as nn
import torch
import math
import timm
import numpy as np
from sklearn import preprocessing
from transformers import CLIPProcessor, CLIPVisionModel, BertTokenizer, LxmertTokenizer, ViTFeatureExtractor, DeiTFeatureExtractor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torchvision import models
import torch.nn.functional as F
from swin_modeling_bert import BertConfig, BertModel, BertOnlyMLMHead
from modeling_lxmert import LxmertConfig, LxmertXLayer
from transformers import DeiTFeatureExtractor, DeiTModel, DeiTForImageClassification, DeiTConfig, SwinForImageClassification


class ContrastProjection(nn.Module):
    def __init__(self, __C):
        super().__init__()
        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 768)

    def forward(self, tokens):
        return self.linear2(F.relu(self.linear1(tokens)))

class Multi_attention_Model(nn.Module):
    def __init__(self, opt, using_amp =False):
        super(Multi_attention_Model, self).__init__()
        self.bias = nn.Parameter(torch.tensor(1),requires_grad = False)
        self.opt = opt
        self.criterion = nn.CrossEntropyLoss()
        self.num_layers = opt.xlayer_num
        self.softmax_image = nn.Softmax(dim=1)
        self.contrast_proj = ContrastProjection(opt).cuda()
        self.min_max_scaler = preprocessing.MinMaxScaler() 
        self.linear = nn.Linear(1000,768)
        self.max_len = opt.max_length
        if opt.dataset == "CUB":
            self.fc_image = nn.Linear(768,312)
        elif opt.dataset == "AWA2":
            self.fc_image = nn.Linear(768,85)
        elif opt.dataset == "SUN":
            self.fc_image = nn.Linear(768,102)  
        elif opt.dataset == "RSSDIVCS" or opt.dataset == "mymars":
            if opt.class_embedding == "w2v":
                self.fc_image = nn.Linear(768,300)
            elif opt.class_embedding == "bert":
                self.fc_image = nn.Linear(768,1024)
        
        self.bert = BertModel.from_pretrained('/work/bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained("/work/bert-base-uncased", do_lower_case=True)

        self.config = BertConfig()
        self.cls = BertOnlyMLMHead(self.config)
        
        if opt.dataset == "AWA2" or opt.dataset == "CUB" or opt.dataset == "RSSDIVCS":
            self.deit = DeiTForImageClassification.from_pretrained("/work/deit-base-distilled-patch16-224")
            # print(self.deit)
            # exit()
        elif opt.dataset == "SUN":
            self.deit = SwinForImageClassification.from_pretrained("/work/swin")


        self.lxmert_config = LxmertConfig()
        self.lxmert_xlayer = LxmertXLayer(self.lxmert_config)

    def forward(self, x, seg_features, texts, whole_semantic=None, con_labels=[],mask_indexs = None, batch_target=None,semantic_deal=None,do_predict=False,texts_label=[],impath=[],texts_label_withpro=[]):
        image_embedding = self.linear(self.deit(x).logits).unsqueeze(1)  

        inputs = self.tokenizer.batch_encode_plus(
            texts,
            padding=True,
            max_length = self.max_len,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids = True,
            return_attention_mask = True,
            add_special_tokens=True
        )  
        label = inputs.input_ids.cuda()

        inputs.attention_mask = inputs.attention_mask.cuda() 
        inputs.input_ids = inputs.input_ids.cuda()      
        inputs.token_type_ids = inputs.token_type_ids.cuda()

        text_embedding = self.bert(
            input_ids=inputs.input_ids,
            token_type_ids=inputs.token_type_ids,
            attention_mask = inputs.attention_mask,
        )
        text_hidden_state = text_embedding[0] 
        text_pool_output = text_embedding[1] 

        lang_feats = text_hidden_state   
        visual_feats = image_embedding
 

        for i in range(self.num_layers):
            x_outputs = self.lxmert_xlayer(
                lang_feats = lang_feats ,
                lang_attention_mask = None,  
                visual_feats = visual_feats,
                visual_attention_mask = None,
                input_id = inputs.input_ids,
                output_attentions=False,
            )
            lang_feats, visual_feats = x_outputs[:2]

        loss_mask = 0.      
    

        image_embedding_class = visual_feats[:, 0, :]
        pre_semantic = self.fc_image(image_embedding_class)
        output_class_image = self.softmax_image(pre_semantic.mm(seg_features))


        if whole_semantic != None and self.opt.sc_loss > 0:
            mask_bias = np.ones((1,whole_semantic.shape[1]))
            mask_bias[:,self.opt.data.seenclasses.cpu().numpy()] *= -1
            self.mask_bias = nn.Parameter(torch.tensor(mask_bias).float(),requires_grad = False).cuda()

            embedding_for_sc = pre_semantic.mm(whole_semantic)
            embedding_for_sc = embedding_for_sc + self.mask_bias*self.bias
        else:
            embedding_for_sc = 0

        return output_class_image, pre_semantic, 0, 0, loss_mask, image_embedding_class, embedding_for_sc

    def info_nce_loss(self, features, con_labels):
        con_labels = (con_labels.unsqueeze(0) == con_labels.unsqueeze(1)).float().cuda()
        con_mask = torch.eye(con_labels.shape[0], dtype=torch.bool).cuda()
        con_labels = con_labels[~con_mask].view(con_labels.shape[0], -1)
        con_labels = (con_labels - 1) * (-1)
        con_labels = torch.cat((con_labels,con_labels),1)
        con_labels_whole = torch.cat((con_labels,con_labels),0)
            
        labels = torch.cat([torch.arange(self.opt.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().cuda()
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        negatives_surcon = negatives * con_labels_whole

        logits = torch.cat([positives, negatives_surcon], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        logits = logits / self.opt.temperature
        return logits, labels
