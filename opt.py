import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='RSSDIVCS', help='FLO, CUB')
    parser.add_argument('--root', default='/work/tanxm/duet2my/', help='path to project')      # 根据文件夹不同需要更改
    parser.add_argument('--image_root', default='/work/tanxm/data/', type=str, metavar='PATH',
                        help='path to image root')
    parser.add_argument('--matdataset', default=True, help='Data in matlab format')
    parser.add_argument('--image_embedding', default='res18')
    parser.add_argument('--class_embedding', default='w2v')
    parser.add_argument('--preprocessing', action='store_true', default=True,
                        help='enbale MinMaxScaler on visual features')
    parser.add_argument('--standardization', action='store_true', default=False)
    parser.add_argument('--ol', action='store_true', default=False,
                        help='original learning, use unseen dataset when training classifier')
    parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=30000, help='number of epochs to train for')
    parser.add_argument('--classifier_lr', type=float, default=1e-6, help='learning rate to train softmax classifier')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
    parser.add_argument('--manualSeed', type=int, default=None, help='manual seed 3483')
    # parser.add_argument('--imagelist', default=CC_HOME + '/ZSL_REG/data/CUB/cub_imagelist.txt', type=str,
    #                     metavar='PATH',
    #                     help='path to imagelist (default: none)')
    parser.add_argument('--resnet_path', default='/work/tanxm/plms/resnet/resnet18-f37072fd.pth',
                        # resnet101_cub.pth.tar resnet101-5d3b4d8f.pth
                        help="path to pretrain resnet classifier")

    parser.add_argument('--train_id', type=int, default=0)
    parser.add_argument('--pretrained', default=None, help="path to pretrain classifier (to continue training)")
    # parser.add_argument('--checkpointroot', default=CC_HOME + '/ZSL_REG/checkpoint', help='path to checkpoint')
    parser.add_argument('--image_type', default='test_unseen_loc', type=str, metavar='PATH',
                        help='image_type to visualize, usually test_unseen_small_loc, test_unseen_loc, test_seen_loc')
    parser.add_argument('--pretrain_epoch', type=int, default=5)
    parser.add_argument('--pretrain_lr', type=float, default=1e-4, help='learning rate to pretrain model')
    parser.add_argument('--all', action='store_true', default=False)

    parser.add_argument('--gzsl', action='store_true', default=False)
    parser.add_argument('--additional_loss', action='store_true', default=True)

    parser.add_argument('--xe', type=float, default=0)
    parser.add_argument('--semantic', type=float, default=0)
    parser.add_argument('--regular', type=float, default=0)
    parser.add_argument('--l_xe', type=float, default=0)
    parser.add_argument('--l_semantic', type=float, default=0)
    parser.add_argument('--l_regular', type=float, default=0)
    parser.add_argument('--cpt', type=float, default=0)
    parser.add_argument('--calibrated_stacking', type=float, default=False,
                        help='calibrated_stacking, shrinking the output score of seen classes')

    # for distributed loader
    parser.add_argument('--train_mode', type=str, default='random', help='loader: random or distributed')
    parser.add_argument('--n_batch', type=int, default=1000, help='batch numbers per epoch')
    parser.add_argument('--ways', type=int, default=16, help='class numbers per episode')
    parser.add_argument('--shots', type=int, default=2, help='image numbers per class')

    parser.add_argument('--transform_complex', action='store_true', default=False, help='complex transform')
    # evaluation
    parser.add_argument('--only_evaluate', action='store_true', default=False)
    parser.add_argument('--resume', default=False)

    
    parser.add_argument('--patient', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--loss_function', type=str, default="L2", help="the type of loss function")
    parser.add_argument('--model_name', type=str, default="baseline", help="baseline,softmax,sigmoid,relu")
    parser.add_argument('--mask_pro', type=float, default=0.5, help="probablity of mask")
    parser.add_argument('--mask_loss_xishu', type=float, default=1, help="The coefficient of mask loss")
    parser.add_argument('--construct_loss_weight', type=float, default=0.01, help="weight of construct_loss")
    parser.add_argument('--xlayer_num', type=int, default=1, help="number of xlayer(cross and self attention)")
    parser.add_argument('--sc_loss', type=float, default=0)
    parser.add_argument('--semantic_miss', type=float, default=0.5)
    parser.add_argument('--gradient_time', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=150)

    # tan添加
    parser.add_argument('--seen_unseen_ratio', type=int, default=6010)
    parser.add_argument('--random_num', type=int, default=1)

    opt = parser.parse_args()
    opt.dataroot = opt.root + 'data'
    opt.checkpointroot = opt.root + 'checkpoint'
    print('opt:', opt)
    return opt
