
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import torch.utils.data
import os
from PIL import Image
import numpy as np
import h5py
import torch
import torch.utils.data
import scipy.io as sio

import json
from transformers import ViTFeatureExtractor, CLIPProcessor, CLIPVisionModel
from transformers import DeiTFeatureExtractor, DeiTModel, DeiTForImageClassification
from transformers import BeitFeatureExtractor, AutoFeatureExtractor



def sort_acc(acc, name):
    sorted_index = sorted(range(len(acc)), key=lambda k: acc[k], reverse=False)
    name_sort = []
    for i in range(len(name)):
        name_sort.append(name[sorted_index[i]])
    acc.sort()
    return acc, name_sort
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def L2_normalize(array):
    # L2 normalize
    norm = np.linalg.norm(array)
    array = array / norm
    return array

def map_label(label, classes):
    mapped_label = torch.LongTensor(len(label))
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    
    return mapped_label

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename+'.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename+'.log', "a")
        f.write(message)  
        f.close()

class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
                
    def read_h5dataset(self, opt):
        # read image feature
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".hdf5", 'r')
        feature = fid['feature'][()]
        label = fid['label'][()] 
        trainval_loc = fid['trainval_loc'][()]
        train_loc = fid['train_loc'][()] 
        val_unseen_loc = fid['val_unseen_loc'][()] 
        test_seen_loc = fid['test_seen_loc'][()] 
        test_unseen_loc = fid['test_unseen_loc'][()] 
        fid.close()
        # read attributes
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".hdf5", 'r')
        self.attribute = fid['attribute'][()]
        fid.close()

        if not opt.validation:
            self.train_feature = feature[trainval_loc] 
            self.train_label = label[trainval_loc] 
            self.test_unseen_feature = feature[test_unseen_loc] 
            self.test_unseen_label = label[test_unseen_loc] 
            self.test_seen_feature = feature[test_seen_loc] 
            self.test_seen_label = label[test_seen_loc] 
        else:
            self.train_feature = feature[train_loc] 
            self.train_label = label[train_loc] 
            self.test_unseen_feature = feature[val_unseen_loc] 
            self.test_unseen_label = label[val_unseen_loc] 

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.nclasses = self.seenclasses.size(0)

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()


        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        if opt.class_embedding == "w2v":
            self.semantic = torch.from_numpy(matcontent['w2v']).float()
        elif opt.class_embedding == "bert":
            self.semantic = torch.from_numpy(matcontent['bert']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long() 
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long() 
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long() 
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)

    def read_matdataset(self, opt):

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['visual_features']
        self.image_features = feature

        self.label = matcontent['visual_classids'].astype(int).squeeze() - 1   
        self.image_files = matcontent['visual_imagefiles'].squeeze()
        self.image_ids = matcontent['visual_imageids'].astype(int).squeeze() - 1        

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + "splits2.mat")

        
        if opt.gzsl == 'gzsl' :
            if opt.seen_unseen_ratio == 6010:
                    self.seen_loc_20 = matcontent['seen_loc1']     
                    self.train_loc_20 = matcontent['train_seen_loc1']     
                    self.test_seen_loc_20 = matcontent['test_seen_loc1']     
                    self.unseen_loc_20 = matcontent['unseen_loc1']     
                    self.test_unseen_loc_20 = self.unseen_loc_20
            elif opt.seen_unseen_ratio == 5020:
                    self.seen_loc_20 = matcontent['seen_loc2']     
                    self.train_loc_20 = matcontent['train_seen_loc2']     
                    self.test_seen_loc_20 = matcontent['test_seen_loc2']     
                    self.unseen_loc_20 = matcontent['unseen_loc2']     
                    self.test_unseen_loc_20 = self.unseen_loc_20                
            elif opt.seen_unseen_ratio == 4030:
                    self.seen_loc_20 = matcontent['seen_loc3']     
                    self.train_loc_20 = matcontent['train_seen_loc3']     
                    self.test_seen_loc_20 = matcontent['test_seen_loc3']    
                    self.unseen_loc_20 = matcontent['unseen_loc3']     
                    self.test_unseen_loc_20 = self.unseen_loc_20


            self.seen_loc = self.seen_loc_20[:,opt.random_num-1].squeeze() - 1    
            self.train_loc = self.train_loc_20[:,opt.random_num-1].squeeze() - 1    
            self.test_seen_loc = self.test_seen_loc_20[:,opt.random_num-1].squeeze() - 1   
            self.unseen_loc = self.unseen_loc_20[:,opt.random_num-1].squeeze() - 1 
            self.test_unseen_loc = self.unseen_loc

        else :
            if opt.seen_unseen_ratio == 6010:
                    self.seen_loc_20 = matcontent['seen_loc1']     
                    self.train_loc_20 = self.seen_loc_20     
                    self.unseen_loc_20 = matcontent['unseen_loc1']    
                    self.test_unseen_loc_20 = self.unseen_loc_20
            elif opt.seen_unseen_ratio == 5020:
                    self.seen_loc_20 = matcontent['seen_loc2']    
                    self.train_loc_20 = self.seen_loc_20     
                    self.unseen_loc_20 = matcontent['unseen_loc2']     
                    self.test_unseen_loc_20 = self.unseen_loc_20                
            elif opt.seen_unseen_ratio == 4030:
                    self.seen_loc_20 = matcontent['seen_loc3']     
                    self.train_loc_20 = self.seen_loc_20     
                    self.unseen_loc_20 = matcontent['unseen_loc3']    
                    self.test_unseen_loc_20 = self.unseen_loc_20


            self.seen_loc = self.seen_loc_20[:,opt.random_num-1].squeeze() - 1    
            self.train_loc = self.train_loc_20[:,opt.random_num-1].squeeze() - 1     
            self.unseen_loc = self.unseen_loc_20[:,opt.random_num-1].squeeze() - 1    
            self.test_unseen_loc = self.unseen_loc
        

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + "images.mat")
        self.class_name = matcontent['class_name']   

        if opt.class_embedding == "w2v":
            semantic = "word2vec.mat"
        elif opt.class_embedding == "bert":
            semantic = "bert2.mat"
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + semantic)
        self.seg_features = torch.from_numpy(matcontent['seg_features']).float()   
        self.seg_class = matcontent['seg_class'].squeeze()      
        

        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                if opt.gzsl == "gzsl":
                    _train_feature = scaler.fit_transform(feature[self.train_loc])
                    _test_seen_feature = scaler.transform(feature[self.test_seen_loc])
                else:
                    _train_feature = scaler.fit_transform(feature[self.seen_loc])        
                _test_unseen_feature = scaler.transform(feature[self.unseen_loc])

                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                if opt.gzsl == "gzsl":
                    self.train_label = torch.from_numpy(self.label[self.train_loc]).long()

                    self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                    self.test_seen_feature.mul_(1/mx)
                    self.test_seen_label = torch.from_numpy(self.label[self.test_seen_loc]).long()
                else :
                    self.train_label = torch.from_numpy(self.label[self.seen_loc]).long()
                    
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(self.label[self.unseen_loc]).long()                      

            else:
                self.train_feature = torch.from_numpy(feature[self.seen_loc]).float()
                self.train_label = torch.from_numpy(self.label[self.train_loc]).long()

                self.test_unseen_feature = torch.from_numpy(feature[self.unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(self.label[self.unseen_loc]).long()

                self.test_seen_feature = torch.from_numpy(feature[self.seen_loc]).float()
                self.test_seen_label = torch.from_numpy(self.label[self.test_seen_loc]).long()
        else:
            if opt.gzsl == "gzsl":
                self.train_feature = torch.from_numpy(feature[self.train_loc]).float()
                self.train_label = torch.from_numpy(self.label[self.train_loc]).long()
            else :
                self.train_feature = torch.from_numpy(feature[self.seen_loc]).float()
                self.train_label = torch.from_numpy(self.label[self.seen_loc]).long()        
            self.test_unseen_feature = torch.from_numpy(feature[self.unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(self.label[self.unseen_loc]).long()

        if opt.gzsl == "gzsl":
            self.seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))
        else :
            self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))


        self.ntrain = self.train_feature.size()[0]      
        self.ntest_unseen = self.test_unseen_feature.size()[0]  
        if opt.gzsl == "gzsl":
            self.ntest_seen = self.test_seen_feature.size()[0]  
        self.ntrain_class = self.seenclasses.size(0)    
        self.ntest_class = self.unseenclasses.size(0)   
        self.train_class = self.seenclasses.clone()     
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 
        
    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0 
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.semantic[iclass_label[0:batch_size]] 
    
    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.semantic[batch_label]
        return batch_feature, batch_label, batch_att


    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]
            
        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))       
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.semantic.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.semantic[batch_label[i]] 
        return batch_feature, batch_label, batch_att

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(opt, image_files, img_loc, image_labels, dataset):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    image_files = image_files[img_loc]
    image_labels = image_labels[img_loc]
    if opt.dataset == 'RSSDIVCS':
        for image_file, image_label, img_lo in zip(image_files, image_labels, img_loc):
            image_file = opt.image_root + 'RSSDIVCS' + str(image_file).split("RSSDIVCS")[2].replace("\\\\","/").split("'")[0]

            text_matrix = opt.semantic_feature[image_label]
            imlist.append((image_file, int(image_label), text_matrix, int(img_lo)))

        return imlist

class ImageFilelist(torch.utils.data.Dataset):
    def __init__(self, opt, data_inf=None, transform=None, target_transform=None, dataset=None,
                 flist_reader=default_flist_reader, loader=default_loader, image_type=None, select_num=None):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.opt = opt

        if opt.model_name == "Multi_attention_Model":
            if opt.dataset == "AWA2" or opt.dataset == "CUB":
                self.feature_extractor = DeiTFeatureExtractor.from_pretrained("/work/tanxm/plms/deit-base-distilled-patch16-224")
            elif opt.dataset == "SUN":
                self.feature_extractor = AutoFeatureExtractor.from_pretrained("/work/tanxm/plms/swin")
            elif opt.dataset == 'RSSDIVCS':
                self.feature_extractor = DeiTFeatureExtractor.from_pretrained("/work/tanxm/plms/deit-base-distilled-patch16-224")
  

        if image_type == 'test_unseen_small_loc':
            self.img_loc = data_inf.test_unseen_small_loc
        elif image_type == 'test_unseen_loc':
            self.img_loc = data_inf.test_unseen_loc
        elif image_type == 'trainval_loc':
            self.img_loc = data_inf.trainval_loc
        elif image_type == 'train_loc':
            self.img_loc = data_inf.train_loc
        elif opt.gzsl == 'gzsl' and image_type == 'test_seen_loc':
            self.img_loc = data_inf.test_seen_loc
        else:
            try:
                sys.exit(0)
            except:
                print("choose the image_type in ImageFileList")

        if select_num != None:

            np.random.shuffle(self.img_loc)
            self.img_loc = self.img_loc[:select_num]

        self.image_files = data_inf.image_files
        self.image_labels = data_inf.label
        self.dataset = dataset
        self.imlist = flist_reader(opt, self.image_files, self.img_loc, self.image_labels, self.dataset)
        self.class_name = data_inf.class_name
        self.seg_class = data_inf.seg_class

        self.image_labels = self.image_labels[self.img_loc]
        label, idx = np.unique(self.image_labels, return_inverse=True)
        self.image_labels = torch.tensor(idx)



    def __getitem__(self, index):
        impath, target, matrix,img_loc = self.imlist[index]
        img = self.loader(impath)

        if self.opt.model_name == "lxmert":
            if self.transform is not None:
                inputs = self.transform(img)
        else:
            inputs = self.feature_extractor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target, impath, matrix, img_loc


    def __len__(self):
        num = len(self.imlist)
        return num

def compute_per_class_acc(test_label, predicted_label, nclass):
    test_label = np.array(test_label)
    predicted_label = np.array(predicted_label)
    acc_per_class = []
    acc = np.sum(test_label == predicted_label) / len(test_label)
    for i in range(len(nclass)):
        idx = (test_label == i)
        acc_per_class.append(np.sum(test_label[idx] == predicted_label[idx]) / np.sum(idx))
    return acc, sum(acc_per_class)/len(acc_per_class)

def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
        acc_per_class = []

        acc = np.sum(test_label == predicted_label) / len(test_label)
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class.append(np.sum(test_label[idx] == predicted_label[idx]) / np.sum(idx))
        return acc, sum(acc_per_class)/len(acc_per_class)


def prepare_semantic_label(semantic, classes):

    classes_dim = classes.size(0)

    semantic_dim = semantic.shape[1]   

    output_semantic = torch.FloatTensor(classes_dim, semantic_dim)
    for i in range(classes_dim):
        output_semantic[i] = semantic[classes[i]]
    return torch.transpose(output_semantic, 1, 0)

def save_correct_imgs(test_label, predicted_label, img_paths, img_locs):
    test_label = np.array(test_label)
    print('len(test_label:', len(test_label))
    predicted_label = np.array(predicted_label)
    correct_idx  = [i for i in range(len(test_label)) if  test_label[i]== predicted_label[i]]
    acc = len(correct_idx) / len(test_label)
    print('correct_impaths', len(correct_idx))

    correct_impaths = [img_paths[correct_idx[i]] for i in range(len(correct_idx))]
    correct_imlocs = [img_locs[correct_idx[i]] for i in range(len(correct_idx))]
    print('correct_impaths', correct_impaths[0])
    print('correct_imlocs', len(correct_imlocs))

    print('overall acc:', acc)
    return correct_imlocs, correct_impaths

def save_predict_txt(output, predict_txt_path, seg_features, predicted_label, target, class_attribute, seg_class, class_names):
    file = open(predict_txt_path, 'w')
    output = output.cpu().numpy()[0]
    predicted_class = [class_names[index][0][0] for index in np.argsort(output)[::-1]]
    file.write('predicted classes:{}\n'.format(predicted_class))
    file.write('predict class:{}\n'.format(class_names[predicted_label][0][0]))
    file.write('ground truth class:{}\n'.format(class_names[target][0][0]))
    predict_semantic_mul = seg_features * class_attribute[predicted_label][0]

    target_semantic_mul = seg_features * class_attribute[target][0]
    predict_semantic_idx = np.argsort(predict_semantic_mul)[::-1]
    target_semantic_idx = np.argsort(target_semantic_mul)[::-1]
    file.write('predict semantic_name for {}, '.format(class_names[predicted_label][0][0]) + 'the value is {}:\n \n'.format(sum(predict_semantic_mul)))
    for idx in predict_semantic_idx:
        file.write('semantic_name:{}, semantic_mul_value:{:.4f}\n'.format(seg_class[idx].strip(), predict_semantic_mul[idx]))
    file.write('\n')
    file.write('predict semantic_name for {}, '.format(class_names[target][0][0]) + 'the value is {}:\n \n'.format(sum(target_semantic_mul)))

    for idx in target_semantic_idx:
        file.write('seg_class:{}, semantic_mul_value:{:.4f}\n'.format(seg_class[idx].strip(), target_semantic_mul[idx]))
    file.write('\n')
    file.close()

def add_paths(opt, img_path, correct, predicted_label, seg_class, class_name, semantic_id=None, semantic_rank=None, semantic_weight = None, dataset = None):
    if dataset == "CUB":
        img_path = img_path.split('images/')[1]
    elif dataset == 'AWA1':
        img_path = img_path.split('JPEGImages/')[1]
    elif dataset == 'RSSDIVCS':
        img_path = img_path.split('JPEGImages/')[1]
    sub_path = img_path.split('/')[0]
    mask_path = os.path.join(opt.image_root, 'visual/{}/{}/masks/'.format(opt.vis_type, opt.train_id))
    mask_path = os.path.join(mask_path, sub_path)
    vis_path = mask_path.replace('masks', '{}_attri_images'.format(opt.image_type.strip('_loc')))
    final_vis_path = mask_path.replace('masks', '{}_all_images'.format(opt.image_type.strip('_loc')))
    raw_path = mask_path.replace('masks', 'raw_images')
    predict_txt_path = raw_path.replace('raw_images', 'predict_results')
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
    if not os.path.exists(predict_txt_path):
        os.makedirs(predict_txt_path)
    if not os.path.exists(final_vis_path):
        os.makedirs(final_vis_path)
    final_mask_name = img_path.split('/')[1].replace('.jpg', '.pkl')
    final_mask_path = os.path.join(mask_path, final_mask_name)
    mask_name = final_mask_name.replace('.pkl', '_{}_{}_{}.pkl'.format(semantic_rank, semantic_id, semantic_weight))
    mask_path = os.path.join(mask_path, mask_name)

    final_vis_name = img_path.split('/')[1].replace('.jpg', '_all.jpg')
    final_vis_path = os.path.join(final_vis_path, correct + "_" + class_name[predicted_label][0][0] + "_" + final_vis_name).replace('_{}.png'.format(semantic_id), '_all.png')
    vis_name = img_path.split('/')[1].replace('.jpg', '_rank{}_{}_weight{:.3}.jpg'.format(semantic_rank, seg_class[semantic_id].strip(), semantic_weight))
    vis_path = os.path.join(vis_path, correct + "_" + class_name[predicted_label][0][0] + "_" + vis_name)
    raw_path = os.path.join(raw_path, correct + "_" + class_name[predicted_label][0][0] + "_" + final_vis_name)
    predict_txt_path = os.path.join(predict_txt_path, correct + "_" + class_name[predicted_label][0][0] + "_" + final_vis_name.replace('.jpg', '.txt'))

    return predict_txt_path, mask_path, vis_path, raw_path, final_mask_path, final_vis_path

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def average(x):
    return x / np.sum(x, axis=0)

def per_dim_dis(A, B):
    """
    input: A, B with size of N*1
    purpose: calculate the per_dim distance of A and B
    :return: dis with same size as A
    """
    dis = np.abs(A - B)
    return dis

def add_image_attri_L2_path(opt, img_path, correct, predicted_label, seg_class, class_name, semantic_id=None, semantic_rank=None, semantic_weight=None, attri_dis=None, dataset = None):
    """ This path is divided by image class.
    Args:
        semantic_id: the id of this attribute
        semantic_rank: The rank of the attribute distance of this attri, 0 means the distance is smallest
        semantic_weight: The distance

    Returns:
        The paths.
    """
    if dataset == "CUB":
        img_path = img_path.split('images/')[1]
    elif dataset == 'AWA1':
        img_path = img_path.split('JPEGImages/')[1]

    mask_path = os.path.join(opt.image_root, 'visual/{}/{}/masks/{}'.format(opt.vis_type, opt.train_id, img_path.split('/')[0]))

    vis_path = mask_path.replace('masks', '{}_attri_images'.format(opt.image_type.strip('_loc')))

    final_vis_path = mask_path.replace('masks', '{}_all_images'.format(opt.image_type.strip('_loc')))
    raw_path = mask_path.replace('masks', 'raw_images')

    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
    if not os.path.exists(final_vis_path):
        os.makedirs(final_vis_path)
    final_mask_name = img_path.split('/')[1].replace('.jpg', '.pkl')
    final_mask_path = os.path.join(mask_path, final_mask_name)
    mask_name = final_mask_name.replace('.pkl', '_{}_{}.pkl'.format(semantic_id, semantic_weight))
    mask_path = os.path.join(mask_path, mask_name)
    final_vis_name = img_path.split('/')[1].replace('.jpg', '_all.jpg')
    final_vis_path = os.path.join(final_vis_path, correct + "_" + class_name[predicted_label] + "_" + final_vis_name).replace('_{}.png'.format(semantic_id), '_all.png')
    vis_name = img_path.split('/')[1].replace('.jpg', '_rank{}_{}_dis{:.3}_weight{:.3}.jpg'.format(semantic_rank, seg_class[semantic_id].strip(), attri_dis, semantic_weight))
    vis_path = os.path.join(vis_path, "{}_pred_{}_GT_{}". format(correct, class_name[predicted_label], vis_name))
    raw_path = os.path.join(raw_path, "{}_pred_{}_GT_{}". format(correct, class_name[predicted_label], final_vis_name))

    final_mask_path_pos = final_mask_path.replace('.pkl', '_pos.pkl')
    final_mask_path_neg = final_mask_path.replace('.pkl', '_neg.pkl')

    final_vis_path_pos = final_vis_path.replace('.jpg', '_pos.jpg')
    final_vis_path_neg = final_vis_path.replace('.jpg', '_neg.jpg')

    return mask_path, vis_path, raw_path, final_mask_path, final_vis_path, final_mask_path_pos, final_vis_path_pos, final_mask_path_neg, final_vis_path_neg

def get_group(fn):
    group_data = np.loadtxt(fn)
    num = int(max(group_data))
    groups = [[] for _ in range(num)]
    for i, id in enumerate(group_data):
        groups[int(id)-1].append(i)
    return groups

def add_glasso(var, group):
    return var[group, :].pow(2).sum(dim=0).add(1e-8).sum().pow(1/2.)

def add_dim_glasso(var, group):
    loss = var[group, :].pow(2).sum(dim=1).add(1e-8).pow(1/2.).sum()
    return loss

