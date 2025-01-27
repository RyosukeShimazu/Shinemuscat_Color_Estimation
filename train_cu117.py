import argparse
import glob
import logging
import os
import random
import time
# from io import BytesIO

# albumentations(2022/5/24)
# from typing import Concatenate
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
import cv2
import matplotlib.pyplot as plt
#from torch_ort import ORTModule
import numpy as np
import openpyxl
import pandas as pd
import torch
import timm # 2022/11/08 for ViT
# from torch.utils.tensorboard import SummaryWriter
#import tensorflow as tf
#import tensorboard as tb
#tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from PIL import Image
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.utils.accuracy_calculator import \
    AccuracyCalculator
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from torch import optim
import torch_optimizer as torch_optim #2022/11/25追加
from adabelief_pytorch import AdaBelief #2022/11/25追加
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler #2022/12/16追加
from torchinfo import summary

# from dataset import ColorDataset
# from ModelClass import EfficientNetb0, ResNet34, ResNet50
# from model import ColorModel
from utils import Averager
# from utils import Averager, image_transform
# from utils import Averager, image_converter, image_transform
import csv
import io
import ast          #astはPython標準ライブラリに含まれているのでインストール不要
from torch.onnx import export

# new_seed
def torch_fix_seed(seed=1234):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


logging.basicConfig(
    format='[%(asctime)s] [%(filename)s]:[line:%(lineno)d] [%(levelname)s] %(message)s', level=logging.INFO)


def filename2age(filename):
    #filename→/misc/Work20/amatatsu/color_leow_tume/data/tume-color_train_dataset_2022-09-02-data/3.5_232
    filename = os.path.basename(filename)
    #filename→3.5_232
    tempstrs = filename.split('_')
    #tempstrs→['3.5', '232']
    return float(tempstrs[0])               #1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5のいずれかを返す。


def filename2age_old(filename):
    #filename→/misc/Work20/amatatsu/color_leow_tume/data/tume-color_train_dataset_2022-07-01-data/4.5_447/1-447-21_4.5_2.jpg
    filename = os.path.basename(filename)
    #filename→1-447-21_4.5_2.jpg
    tempstrs = filename.split('_')
    return float(tempstrs[1])   #tempstrs[1]→4.5


def augment(x, data, tume, aug):
# def augment(x, data, aug):
    data_90 = cv2.rotate(data, cv2.ROTATE_90_CLOCKWISE)         #cv2.rotate()関数は、画像を回転するための関数。元の画像を時計回りに90度回転した新しい配列であるdata_90を作成。
    x.append([data_90, tume])
    # x.append(data_90)
    data_270 = cv2.rotate(data, cv2.ROTATE_90_COUNTERCLOCKWISE) #元の画像を反時計回りに90度回転した新しい配列であるdata_270を作成。
    x.append([data_270, tume])
    # x.append(data_270)
    if aug == 8:
        data_180 = cv2.rotate(data, cv2.ROTATE_180)             #元の画像を180度回転した新しい配列であるdata_180を作成。
        x.append([data_180, tume])
        # x.append(data_180)
    return x


def get_data(text_file, path="", path2="", path3="", path4=""):
    assert path != "","blank path name"
    assert type(path) == str,"please use a str input path"

    box = {1.5:[], 2.0:[], 2.5:[], 3.0:[], 3.5:[], 4.0:[], 4.5:[]}

    #粒画像のデータを取得
    data1 = sorted(glob.glob(f'{path}/*'))  #dataの取得
    data2 = sorted(glob.glob(f'{path2}/*')) #dataの取得
    data3 = sorted(glob.glob(f'{path3}/*')) #dataの取得
    data4 = sorted(glob.glob(f'{path4}/*')) #dataの取得
    data = data1 + data2 + data3 + data4
    #ノイズのデータを除外
    with open(text_file, 'r') as f:
        paths_in_text = [line.strip() for line in f.readlines()]
    data = [bunch_path for bunch_path in data if bunch_path in paths_in_text]

    data = random.sample(data, len(data))   #dataをシャッフル

    for d in data:  #d →「/misc/Work20/amatatsu/color_leow_tume/data/tume-color_train_dataset_2022-06-02-data/1.5_000」
        box[filename2age(d)].append(d)

    path_dict = {'train':[], 'valid':[]}

    for label, bunch_path_list in box.items():   #b → 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5
        if len(bunch_path_list) == 0:
            continue
        else: 
            train, valid = train_test_split(bunch_path_list, train_size=0.9, random_state=1) #train_test_split関数は、box[b]をトレーニングセットとバリデーションセットに分割するために使用。
                                                                                    #train_size=0.9を指定することにより、トレーニングセットに90%のデータを割り当てる。
                                                                                    #random_state=1を指定することにより、ランダムなデータの分割方法を固定する。このようにすることで、実行するたびに異なる結果が得られることを防止できる。
            path_dict['train'] += train
            path_dict['valid'] += valid
    return path_dict


def data_balance(path_dict_all, arguments): #データ数の少ないラベルの個数を基準に，各ラベルのデータバランスを調整。ここでは，最もデータ数の少ないラベルの個数の1.5倍を許容限度として，各ラベルのデータをランダムにサンプリングする。
    path_dict = {'train':[], 'valid':[]}
    for p in ['train', 'valid']:
        box = {1.5:[], 2.0:[], 2.5:[], 3.0:[], 3.5:[], 4.0:[], 4.5:[]}
        for d in path_dict_all[p]:
            box[filename2age(d)].append(d)
        
        small_count = 9999
        for b in box:
            if len(box[b]) < small_count and len(box[b]) != 0:
                small_count = len(box[b])

        box_new = {1.5:[], 2.0:[], 2.5:[], 3.0:[], 3.5:[], 4.0:[], 4.5:[]}

        for b in box_new:
            if len(box[b]) > round(small_count*float(arguments["balance_rate"])):
                box_new[b] = random.sample(box[b], k=round(small_count*float(arguments["balance_rate"])))
            else:
                box_new[b] = box[b]
            path_dict[p] += box_new[b]
    return path_dict


def image_converter(img, type="LGB"):
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #cv2.cvtColor() 関数を使用して、imgをRGB色空間に変換
    LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  #cv2.cvtColor() 関数を使用して、imgをLAB色空間に変換
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #cv2.cvtColor() 関数を使用して、imgをHSV色空間に変換
    RGB_split = cv2.split(RGB)                  #cv2.split() 関数を使用して、各色空間のチャンネルを個別に取得
    LAB_split = cv2.split(LAB)
    HSV_split = cv2.split(HSV)

    if type == "RGB":
        return cv2.merge((RGB_split[0],RGB_split[1],RGB_split[2]))
    elif type == "LGB":
        return cv2.merge((LAB_split[0],RGB_split[1],RGB_split[2]))
    elif type == "RLB":
        return cv2.merge((RGB_split[0],LAB_split[0],RGB_split[2]))
    elif type == "RGL":
        return cv2.merge((RGB_split[0],RGB_split[1],LAB_split[0]))
    elif type == "VGB":
        return cv2.merge((HSV_split[2],RGB_split[1],RGB_split[2]))
    elif type == "RVB":
        return cv2.merge((RGB_split[0],HSV_split[2],RGB_split[2]))
    elif type == "RGV":
        return cv2.merge((RGB_split[0],RGB_split[1],HSV_split[2]))
    elif type == "LAB":
        return cv2.merge((LAB_split[0],LAB_split[1],LAB_split[2]))


class ColorDataset():
    def __init__(self, X, Y_color, image_type, H, W, trans_tubu=None, phase='train'):
        self.data = X               # 粒画像のpath or 爪画像のpath
        self.target_c = Y_color     # 正解ラベル。[粒の色, 爪の色, embed_encoderの値] or [粒の色, 爪の色]
        self.transform = trans_tubu # transform
        self.phase = phase          # train or valの指定
        self.image_type = image_type
        self.H = H
        self.W = W
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        x = cv2.imread(x)           # 画像の読み込み
        resized = cv2.resize(x.copy(), dsize=(self.H, self.W))                                      # 224x224にリサイズ
        # cropped = resized[int(self.H/2-112):int(self.H/2+112),int(self.W/2-112):int(self.W/2+112)]  # 画像中央の224x224の画像を生成
        x = image_converter(resized, type=self.image_type)                                          # 画像の色空間を変換
        x = Image.fromarray(x)      #NumPy配列xをPIL(Python Imaging Library)のImageオブジェクトに変換
        x = self.transform(x)                                                        # Albumentationsでの変換はNumPy配列を引数にとる。

        y_c = self.target_c[index]  # 正解ラベル

        return x, y_c


def TumeLoss(outputs, targets, tume, targets_tume):
    target_offset = targets - targets_tume              #正解ラベルの差
    tume_offset = outputs - tume                        #推定値の差
    loss = torch.mean((tume_offset-target_offset)**2) # MSE
    return loss


classifier_names = ["Linear","BiLSTM","LSTM","Transformer"]

feature_extractor = {
    #'モデル名':(モデルのインスタンス, 特徴量抽出層の名前, ？)
    "alexnet":(models.alexnet, "features", False), # 1
    'resnet18':(models.resnet18, "layer4", False),
    'resnet34':(models.resnet34, "layer4", False),
    'resnet50':(models.resnet50, "layer4", False),
    'resnet101':(models.resnet101, "layer4", False), # 2
    'resnet152':(models.resnet152, "layer4", False),
    "vgg11" : (models.vgg11, "features", False), # 3
    "vgg13" : (models.vgg13, "features", False), # 3
    "vgg16" : (models.vgg16, "features", False),
    "vgg19" : (models.vgg19, "features", False),
    "squeezenet" : (models.squeezenet1_0, "features", False),
    "googlenet" : (models.googlenet, "inception5b", False),
    "shufflenet" : (models.shufflenet_v2_x1_0, "conv5", False),
    "mobilenet_v2" : (models.mobilenet_v2, "features", False),
    "resnext50_32x4d" : (models.resnext50_32x4d, "layer4", False),
    "wide_resnet50_2" : (models.wide_resnet50_2, "layer4", False),
    "mnasnet" : (models.mnasnet1_0, "layers", False),
    "densenet121":(models.densenet121, "features", False),
    "densenet" : (models.densenet161, "features", False), # 4
    "efficientnet-b0":(EfficientNet.from_pretrained, "", True), # 5
    "efficientnet-b1":(EfficientNet.from_pretrained, "", True),
    "efficientnet-b2":(EfficientNet.from_pretrained, "", True),
    "efficientnet-b3":(EfficientNet.from_pretrained, "", True),
    "efficientnet-b4":(EfficientNet.from_pretrained, "", True),
    "efficientnet-b5":(EfficientNet.from_pretrained, "", True),
    "efficientnet-b6":(EfficientNet.from_pretrained, "", True),
    "efficientnet-b7":(EfficientNet.from_pretrained, "", True),
    
    #ViT(num_classes=0で特徴量抽出部分のみ)
    #https://logmi.jp/tech/articles/325737
    "vit_tiny_patch16_224":(timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0), "head", False),      # vit_tiny_patch16_224
    "vit_small_patch16_224":(timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0), "head", False),    # vit_small_patch16_224
    "vit_small_patch32_224":(timm.create_model('vit_small_patch32_224', pretrained=True, num_classes=0), "head", False),    # vit_small_patch32_224
    "vit_base_patch8_224":(timm.create_model('vit_base_patch8_224', pretrained=True, num_classes=0), "head", False),        # vit_base_patch8_224
    "vit_base_patch16_224":(timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0), "head", False),      # vit_base_patch16_224
    "vit_base_patch32_224":(timm.create_model('vit_base_patch32_224', pretrained=True, num_classes=0), "head", False),      # vit_base_patch32_224
    "vit_large_patch16_224":(timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=0), "head", False),    # vit_large_patch16_224
}

class FeatExt(nn.Module):
    def __init__(self,model,output_layer = None):                           #ex. model = models.resnet18(pretrained=True), output_layer = "layer4"
        super(FeatExt,self).__init__()
        self.pretrained = model                                             #ex. self.pretrained = models.resnet18(pretrained=True)
        self.output_layer = output_layer                                    #ex. self.output_layer = "layer4"
        self.layers = list(self.pretrained._modules.keys())                 #ex. self.layers　=　['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1,len(self.layers)-self.layer_count):                #resnet18の場合、self.layer_count=7 → range(1, 3) → 1, 2
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])  #._modules.pop() を使用することで、シーケンシャルモジュール内の特定のサブモジュールを削除(今回は、分類器部分を削除)
        
        self.net = nn.Sequential(self.pretrained._modules)                  #一連のニューラルネットワークレイヤーを順番に並べて、一つのネットワークモデルを構築。
        self.pretrained = None

    def forward(self,x):
        x = self.net(x)
        return x


class FeatExtractor(nn.Module):
    def __init__(self,feat_ext_name):                                                               #feat_ext_name：特徴量抽出器の名前。(ex."vit_tiny_patch16_224")
        super(FeatExtractor,self).__init__()
        self.feat_ext_name = feat_ext_name                                                          #ex.「self.feat_ext_name　=　"vit_tiny_patch16_224"」
        self.extract_fn = False                                                                     #self.extract_fn=Falseで初期化
        self.feat_ext_fn ,self.layer_name, self.extract_fn = feature_extractor[self.feat_ext_name]  #ex.「self.feat_ext_fn = (timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0), self.layer_name = "head", self.extract_fn = False」
        
        if "efficientnet" in self.feat_ext_name:                                                    #self.feat_ext_name='efficientnet-b?'の場合
            self.feat_ext = self.feat_ext_fn(self.feat_ext_name)                                    #ex.「self.feat_ext = EfficientNet.from_pretrained("efficientnet-b0")」
        
        elif 'vit' in self.feat_ext_name:                    
            self.feat_ext = self.feat_ext_fn                                                        #ex.「self.feat_ext = (timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)」
        
        else:
            if self.layer_name == "features":
                self.feat_ext = self.feat_ext_fn(pretrained=True).features                          #特徴抽出器部分を取得。学習済み重みを使用して初期化
            else:
                self.feat_ext = FeatExt(self.feat_ext_fn(pretrained=True), self.layer_name)         #特徴抽出器部分を取得。学習済み重みを使用して初期化
            
            
    def forward(self,inputs):
        if self.extract_fn:
            return self.feat_ext.extract_features(inputs)
        else:
            return self.feat_ext(inputs)


class Classifier(nn.Module):
    def __init__(self,arguments,input_channel=1280):
        super(Classifier,self).__init__()
        assert arguments["type"] in classifier_names, f'The {arguments["type"]} classifier is not available'
        self.input_channel = input_channel
        if arguments["type"]:
            #分類器の構成
            self.classifier = nn.Sequential(
                nn.Linear(self.input_channel, 1))   #回帰モデルなので、最後の出力層は1
        self.embedding_layers  = None
        # self.mtl = arguments["mtl"] #距離学習フラグがTrueかFalseか
        self.triplet = arguments["triplet"] #距離学習ブランチの有無
        
        if self.triplet:
            self.norm_embed = arguments["l2_norm_embed"]    #l2_norm_embedがTrueの場合、特徴量ベクトルを正規化する
            #Triplet Loss用
            self.embedding_layers = nn.Sequential(
                nn.Linear(self.input_channel, 128)
            )

    def forward(self,inputs):
        if self.triplet:
            regression = self.classifier(inputs)                   #分類器を通す
            embedding_vector = self.embedding_layers(inputs)       #Triple Loss用のネットワークに通す

            if self.norm_embed:
                embedding_vector = F.normalize(embedding_vector, p=2, dim=1)    #引数：embedding_vector: 正規化するテンソル。p: 正規化時に使用するp-normの値。p=2の場合はL2ノルム。dim: 正規化する軸の次元。dim=1の場合は各行を正規化。
                

            return regression,embedding_vector
            
        return self.classifier(inputs)


class ColorModel(nn.Module):
    def __init__(self,arguments):
        super(ColorModel,self).__init__()
        self.feat_ext = FeatExtractor(arguments["feat_type"])   #arguments["feat_type"]：特徴量抽出器の名前。(ex."vit_tiny_patch16_224")
        shape = self.feat_ext(torch.zeros((1,3,224,224))).shape
        # shape:torch.Size([1, 768] )
        # shape[1]:768
        self.classifier = Classifier(arguments, input_channel=shape[1])

    def forward(self,inputs):
        out = self.classifier(self.feat_ext(inputs))
        return out

def main(args):
    currentDateAndTime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")       #現在の日付と時間を取得
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'GPU:{torch.cuda.is_available()}, device:{device}\n')            #「torch.cuda.is_available()」：現在の環境でCUDAが利用可能である場合にTrue、利用できない場合はFalseを返す。

    arguments = vars(args)                                          #vars()関数は、辞書型に変換
    print(f'arguments：{arguments}\n')

    # start a new wandb run to track this script
    wandb.init(                                                     #wandbの実験（run）を開始。ここで設定するパラメータは実験の追跡に用いられる。
            entity=args.wandb_entity,                               #wandb上のチーム名または個人名を指定。この実験がどのエンティティに関連付けられるかを示す。    
            project=args.wandb_project,                             #この実験がログを保存するプロジェクトの名前を指定
            config=arguments                                        #実験の設定やハイパーパラメータを辞書形式で保存
        )
    if args.triplet:
        wandb.run.name = f"{currentDateAndTime}_{args.feat_type}_{args.optimizer}_{args.lr}_{args.image_type}_cost_weight{args.c_rate}-triplet_weight{args.t_rate}-margin{args.margin}-tume_weight{args.tume_rate}_torch-const{args.const}-bright{args.bright}_balance{args.balance_rate}_seed{seed}"   #今回の実験（run）の名前を設定
    else:
        wandb.run.name = f"{currentDateAndTime}_{args.feat_type}_{args.optimizer}_{args.lr}_{args.image_type}_cost_weight{args.c_rate}-triplet_weight{args.t_rate}-tume_weight{args.tume_rate}_torch-const{args.const}-bright{args.bright}_balance{args.balance_rate}_seed{seed}"   #今回の実験（run）の名前を設定
    
    H, W = args.img_size, args.img_size #リサイズ後の画像のサイズ

    path1 = "/misc/Work20/amatatsu/color_leow_tume/data/tume-color_train_dataset_2022-06-02-data"   #6/2撮影データ
    path2 = "/misc/Work20/amatatsu/color_leow_tume/data/tume-color_train_dataset_2022-07-01-data"   #2022/7/1撮影データ
    path3 = "/misc/Work20/amatatsu/color_leow_tume/data/tume-color_train_dataset_2022-09-02-data"   #2022/9/2撮影データ
    path4 = "/misc/Work20/amatatsu/color_leow_tume/data/tume-color_train_dataset"                   #中間までのデータ

    path_dict_all = get_data(text_file='./bunch_path.txt', path=path1, path2=path2, path3=path3, path4=path4) #get_data関数(それぞれのパスのデータを結合、シャッフルし。ラベルごとに訓練：検証 = 9:1に分割)
    path_dict = data_balance(path_dict_all, arguments)                          #data_balance関数(各ラベルのデータバランスを調整)

    # 房個数表示
    for p in ['train', 'valid']:
        box = {1.5:[], 2.0:[], 2.5:[], 3.0:[], 3.5:[], 4.0:[], 4.5:[]}
        print(f'{p} phase')
        for path in path_dict[p]:
            box[filename2age(path)].append(path)
        
        for label in box:
            print(f'{label} husa : {len(box[label])}')

    print(f'trainの房の個数: {len(path_dict["train"])}')
    print(f'validationの房の個数: {len(path_dict["valid"])}')

    # 粒データ個数表示部分
    for p in ['train', 'valid']:
        tubu_count, tume_count = 0, 0
        box_tubu = {1.5:[], 2.0:[], 2.5:[], 3.0:[], 3.5:[], 4.0:[], 4.5:[]}
        for d in path_dict[p]:
            img_dict_nail = sorted(glob.glob(f'{d}/0-*'))   #爪画像のpathを取得
            img_dict = sorted(glob.glob(f'{d}/1-*'))  #img_dict：['/misc/Work20/amatatsu/color_leow_tume/data/tume-color_train_dataset_2022-07-01-data/2.5_074/0-074-28_3.0_1.jpg',　•••,　 '/misc/Work20/amatatsu/color_leow_tume/data/tume-color_train_dataset_2022-07-01-data/2.5_074/1-074-28_2.5_1.jpg']
            #粒画像について、ノイズのデータを除外
            with open('./path.txt', 'r') as f:
                paths_in_text = [line.strip() for line in f.readlines()]
            img_dict = [img_path for img_path in img_dict if img_path in paths_in_text]
            img_dict = img_dict_nail + img_dict   #爪画像を先頭に追加
            for img_path in img_dict:
                filename = os.path.basename(img_path)
                tume_label = filename.split('_')[0].split('-')[0]
                if tume_label == '0': 
                    tume_count += 1
                else: 
                    tubu_count += 1
                    box_tubu[filename2age_old(img_path)].append(img_path)
        print(f'{p} phase')
        print(f'粒の数 : {tubu_count}')
        print(f'爪の数 : {tume_count}')
        for b in box_tubu:
            print(f'{b} tubu : {len(box_tubu[b])}')


    all_data = len(path_dict['train'])+len(path_dict['valid'])

    dataloaders_dict = {'train':[], 'valid':[]}
    dataloaders_dict_tume = {'train':[], 'valid':[]}
    total_data = []

    normalization = args.normalization  #normalization = [1.5, 4.5]
    if args.triplet:
        embed_encoder = {
            "1.5":0,
            "2.0":1,
            "2.5":2,
            "3.0":3,
            "3.5":4,
            "4.0":5,
            "4.5":6,
        }


    if args.image_type == "RGB":            #画像の標準化で使う平均値meanと標準偏差stdの値を指定。(「/misc/Work20/amatatsu/color_leow_tume/mean_std_ext2.py」で算出し、その値をコピペする)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.image_type == "LGB":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.image_type == "RLB":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.image_type == "RGL":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.image_type == "VGB":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.image_type == "RVB":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.image_type == "RGV":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]


    # transformer torchvision(データ拡張処理を設定)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), 
        transforms.ColorJitter(brightness=args.bright, contrast=args.const),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)])

    train_transform_tume = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)])
    
    valid_transform_tume = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)])

    valid_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)])

    image_transform = {
        "train":train_transform,
        "train_tume":train_transform_tume,
        "valid_tume":valid_transform_tume,
        "valid":valid_transform,
    }


    for phase in ['train', 'valid']:
        if phase =='train':
            tubu_transofmer = image_transform["train"]
            tume_transofmer = image_transform["train_tume"]
        else:
            tubu_transofmer = image_transform["valid"]
            tume_transofmer = image_transform["valid_tume"]

        x = []          #粒画像のリスト
        tume= []        #爪画像のリスト
        y_color = []    #
        for i,path in enumerate(path_dict[phase]):
            #path→/misc/Work20/amatatsu/color_leow_tume/data/tume-color_train_dataset/1.5_025
            img_dict_nail = sorted(glob.glob(f'{path}/0-*'))   #爪画像のpathを取得
            img_dict = sorted(glob.glob(f'{path}/1-*')) #粒画像のpathを取得
            #粒画像について、ノイズのデータを除外
            with open('./path.txt', 'r') as f:
                paths_in_text = [line.strip() for line in f.readlines()]
            img_dict = [img_path for img_path in img_dict if img_path in paths_in_text]
            img_dict = img_dict_nail + img_dict   #爪画像を先頭に追加

            for img_path in img_dict:
                #img_path → /misc/Work20/amatatsu/color_leow_tume/data/tume-color_train_dataset_2022-06-02-data/1.5_014/1-014-37_1.5_1.jpg
                filename = os.path.basename(img_path)   #filename → 1-014-37_1.5_1.jpg
                tume_label = filename.split('_')[0].split('-')[0]   #tume_label → 1 or 0(1:粒の画像, 0:爪の画像)

                if tume_label == '0':
                    tume_path = img_path
                else:
                    tubu_path = img_path
                    x.append(tubu_path)
                    tume.append(tume_path)

                    number = filename2age(path)                                 #number→1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5のいずれか

                    # normalization = [1.5, 4.5]
                    number = (number - normalization[0]) / (normalization[1] - normalization[0])    #numberを0~1に正規化(number-1.5/3)
                    tume_number = (3.0 - normalization[0]) / (normalization[1] - normalization[0])  #tume_numberを0~1に正規化(3.0-1.5/3)
        

                    if args.triplet:    #マルチタスク学習(距離学習、爪学習を含める)場合
                        for T in range(args.aug):
                            y_color.append([number, tume_number, embed_encoder[str(filename2age(path))]])   #[粒の色, 爪の色, embed_encoderの値]
                    else:   #シングルタスク学習(色推定単体)の場合
                        for T in range(args.aug):
                            y_color.append([number, tume_number])                                           #[粒の色, 爪の色]
        y_color = torch.tensor(y_color, dtype = torch.float32)
        print(f'{phase}_data: 房数：{len(path_dict[phase]):>5,} 粒数：{len(x):>6,} 粒/房：({len(x)/len(path_dict[phase]):.1f}, 分割比率：{round((len(path_dict[phase])/all_data)*100):>3}%)')   #all_data：len(path_dict['train'])+len(path_dict['valid'])
        total_data.append(len(x))   #total_data→[8081(trainデータの数), 1081(validationデータの数)]
        
        #type(x)→<class 'numpy.ndarray'>
        dataloaders_dict[phase] = DataLoader(ColorDataset(x, y_color, args.image_type, H, W, trans_tubu=tubu_transofmer, phase=phase), batch_size=args.batch_size, collate_fn=None, shuffle=True, num_workers=8, pin_memory=True)          #dataloaderを生成
        dataloaders_dict_tume[phase] = DataLoader(ColorDataset(tume, y_color, args.image_type, H, W, trans_tubu=tume_transofmer, phase=phase), batch_size=args.batch_size, collate_fn=None, shuffle=True, num_workers=8, pin_memory=True)
        #DataLoaderの引数
        #collate_fn：複数のサンプルを1つのバッチにまとめる方法を定義(None場合は、サンプルのリストを単純にまとめたリストを返す)
        #num_workers=8：8つのワーカーを使用してデータを並列に読み込む
        #pin_memory=True：GPUメモリにデータをピン留めすることで、データの読み込みを高速化することができる。


    model = ColorModel(arguments).to(device)    #arguments：vars(args)

    #モデル構造の可視化
    summary(model=model, input_size=(args.batch_size, 3, 224,224))


    #最適化関数の設定
    if args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))  #betas: 重み付き平均を計算する際の係数を指定するタプル。
    elif args.optimizer.lower() == "sgd":     
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=1e-4)
    elif args.optimizer.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99,
                                        eps=1e-08, weight_decay=0, momentum=0.01, centered=False)
    elif args.optimizer.lower() == "lamb":
        optimizer = torch_optim.Lamb(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=1e-4)
    elif args.optimizer.lower() == "adabelief":
        optimizer = AdaBelief(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    elif args.optimizer.lower() == "adabelief_ori":
        optimizer = AdaBelief(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    elif args.optimizer.lower() == "radam":     
        optimizer = torch_optim.RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    elif args.optimizer == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=args.rho, eps=args.eps)
    
    #Schedulerの設定
    #https://katsura-jp.hatenablog.com/entry/2019/01/30/183501
    #https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=5,verbose=True) #lossが減少しなくなったらlrを減少させる

    # 損失関数の設定
    mse_criterion = nn.MSELoss().to(device)
    if args.triplet:
        distance = CosineSimilarity()       #コサイン類似度：2つのベクトルが「どのくらい似ているか」という類似性を表す尺度(具体的にはベクトル空間における2つのベクトルがなす角のコサイン値)
        reducer = ThresholdReducer(low = 0) #指定された範囲内にある損失のみを使用して、平均損失を計算
        triplet_loss_criterion = TripletMarginLoss(margin=args.margin, distance=distance, reducer=reducer)
        mining_func = TripletMarginMiner(margin=args.margin, distance=distance, type_of_triplets="semihard")
        accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    # Color Loss
    color_loss_averager  = {
        "train":Averager(),
        "valid":Averager()
    }
    # Triplet Loss
    triplet_loss_averager  = {
        "train":Averager(),
        "valid":Averager()
    }
    # Tume Loss
    tume_loss_averager  = {
        "train":Averager(),
        "valid":Averager()
    }
    # 全体Loss
    loss_averager  = {
        "train":Averager(), #Averager()：平均を計算するクラス
        "valid":Averager()
    }
    history = {'train_loss': [], 'valid_loss': [], 'train_bestloss': 100, 'valid_bestloss': 100}

    start = time.time() #学習の開始時刻
    
    for epoch in range(args.epochs):                            #学習部分
        for phase in ["train","valid"]:
            model.train() if phase == 'train' else model.eval() # モデルのモード設定
            for tubu, tume in zip(tqdm(dataloaders_dict[phase], position=0, desc=f'Epoch {epoch+1}/{args.epochs}'), tqdm(dataloaders_dict_tume[phase], position=0, desc=f'Epoch {epoch+1}/{args.epochs}')): #zip：複数のイテラブルオブジェクトから同じインデックスの要素を取り出し、それらをタプルにまとめたイテレータを返す関数。
                # tqdmの引数について
                # position: 進捗バーを表示する位置を指定
                # desc: 進捗バーの前に表示される説明を指定

                optimizer.zero_grad()

                inputs = tubu[0]    # inputs.shape:torch.Size([100, 3, 224, 224])。# inputs.reshape(-1).shape: torch.Size([15052800])

                labels = tubu[1]    # label.shape:torch.Size([100, 3])

                tumes = tume[0]     # tumes.shape:torch.Size([100, 3, 224, 224])

                inputs.to(device)
                tumes.to(device)
                labels.to(device)
                with torch.set_grad_enabled(phase == 'train'): # 訓練時は勾配計算、検証時は勾配計算しない(withでメモリ消費を抑えてる？).torch.set_grad_enabled()：自動微分機能を有効または無効にするための関数
                    if args.triplet:
                        outputs, embed_vec = model(inputs.to(device))   #outputs.shape:torch.Size([1, 1])。embed_vec.shape:torch.Size([1, 128])。
                        
                        #MSE Loss
                        cost = mse_criterion(outputs.reshape(-1).to(device), labels[:,0].to(device)) # 色推定Lossの計算
                        color_loss_averager[phase].add(cost.cpu()) #.add()：テンソルなどの要素同士を加算する関数。
                        
                        #Triplet Loss
                        indices_tuple = mining_func(embed_vec, labels[:,2].to(device))                          #TripletMarginMinerの計算
                        triplet_loss = triplet_loss_criterion(embed_vec,labels[:,2].to(device),indices_tuple)   #距離学習Lossの計算
                        triplet_loss_averager[phase].add(triplet_loss.cpu()) #.add()：テンソルなどの要素同士を加算する関数。
                        
                        #Tume Lossを含める場合
                        if args.tume:
                            tume_outputs, embed_vec_tume = model(tumes.to(device))  #tume_outputs.shape:torch.Size([1, 1])

                            tume_loss = TumeLoss(outputs.reshape(-1).to(device), labels[:,0].to(device), tume_outputs.reshape(-1).to(device), labels[:,1].to(device))   #爪学習Lossの計算。outputs.reshape(-1)：outputsの形状を自動的に1次元に変換
                            tume_loss_averager[phase].add(tume_loss.cpu()) #.add()：テンソルなどの要素同士を加算する関数。

                            #全体Loss
                            loss = args.c_rate*cost + args.t_rate*triplet_loss + args.t_rate*tume_loss       #lossの計算
                        else:
                            loss = args.c_rate*cost + args.t_rate*triplet_loss
                    
                    else:
                        outputs = model(inputs.to(device))

                        #MSE Loss
                        cost = mse_criterion(outputs.reshape(-1).to(device), labels[:,0].to(device))
                        color_loss_averager[phase].add(cost.cpu()) #.add()：テンソルなどの要素同士を加算する関数。
                        
                        #Tume Lossを含める場合
                        if args.tume:
                            tume_outputs = model(tumes.to(device))  #tume_outputs.shape:torch.Size([1, 1])

                            tume_loss = TumeLoss(outputs.reshape(-1).to(device), labels[:,0].to(device), tume_outputs.reshape(-1).to(device), labels[:,1].to(device))   #爪学習Lossの計算。outputs.reshape(-1)：outputsの形状を自動的に1次元に変換
                            tume_loss_averager[phase].add(tume_loss.cpu()) #.add()：テンソルなどの要素同士を加算する関数。
                            
                            #全体Loss
                            loss = args.c_rate*cost + args.tume_rate*tume_loss
                        else:
                            loss = cost

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    loss_averager[phase].add(loss.cpu())    #.add()：テンソルなどの要素同士を加算する関数。
            wandb.log({f"{phase}_loss": loss_averager[phase].val()})
            if args.triplet or args.tume:
                wandb.log({f"{phase}_color_loss": color_loss_averager[phase].val()})
                color_loss_averager[phase].reset()
                if args.triplet:
                    wandb.log({f"{phase}_triplet_loss": triplet_loss_averager[phase].val()})
                    triplet_loss_averager[phase].reset()
                if args.tume:
                    wandb.log({f"{phase}_tume_loss": tume_loss_averager[phase].val()})
                    tume_loss_averager[phase].reset()
            
            if phase == "valid":
                val_loss = loss_averager[phase].val()
                scheduler.step(val_loss)    #val_lossが下がらなければ減衰
            
            history[f'{phase}_loss'].append(loss_averager[phase].val()) #lossの追加
            if history[f'{phase}_bestloss'] > loss_averager[phase].val():
                torch.save(model.state_dict(), f'{args.output_dir}/model_best_{phase}.pth') #pthファイルを保存
                history[f'{phase}_bestloss'] = loss_averager[phase].val()
            loss_averager[phase].reset()    #.reset()：loss_averager[phase]を初期状態に戻す

    end = time.time()   #学習の終了時間

    train_bestloss = min(history['train_loss'])
    valid_bestloss = min(history['valid_loss'])

    wandb.log({
            'Train_Bestloss：': train_bestloss, 
            'Validation_Bestloss：': valid_bestloss,
        })
    
    with open(f'{args.output_dir}/result_train.txt', mode='w') as f:
        f.write('【Conditions】\n'+
                f'Model : {args.output_dir}\n'+
                f'Optim : {args.optimizer}(lr={args.lr})\n'+
                f'Epoch : {args.epochs}\n'+
                f'Batch : {args.batch_size}\n'+
                f'Time : {(end-start)/3600:.3f}h\n'+
                f'Train_bestloss : {train_bestloss:.6f}\n'+
                f'Valid_bestloss : {valid_bestloss:.6f}\n\n')
        f.write('【Data_amount】\n')
        for i,phase in enumerate(['train', 'valid']):
            f.write(f'{phase:>5}_data: {len(path_dict[phase]):>5,}→{total_data[i]:>6,} (×{total_data[i]/len(path_dict[phase])}, {round((len(path_dict[phase])/all_data)*100):>3}%)\n')  #all_data：len(path_dict['train'])+len(path_dict['valid'])
        f.write('-'*40+f'\ntoall_data: {all_data:>5,}→{sum(total_data):>6,} (×{sum(total_data)/all_data}, 100%)\n') #total_data：全体の粒の数


    #学習済みモデルを読み込む
    model.load_state_dict(torch.load(f'{args.output_dir}/model_best_valid.pth'))
    model.eval()
    #クラス毎の精度表示
    number = ['1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5']
    embed_lab = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    embedvec_box_eval, embedlab_box_eval =  [],[]

    for phase in ['valid']:
        outbox = {'1.5':[], '2.0':[], '2.5':[], '3.0':[], '3.5':[], '4.0':[], '4.5':[]} #出力値を格納する箱
        acc_half = {'1.5':0, '2.0':0, '2.5':0, '3.0':0, '3.5':0, '4.0':0, '4.5':0}      #ラベルが±0.5以内に入ってるもの
        acc_quarter = {'1.5':0, '2.0':0, '2.5':0, '3.0':0, '3.5':0, '4.0':0, '4.5':0}   #ラベルが±0.25以内に入ってるもの
        label, output, mean, median, sd, acc_h, acc_q = [],[],[],[],[],[],[] 
        logging.info(f"evaluating phase:{phase}")
        with torch.no_grad():
            for tubu,tume in zip(dataloaders_dict[phase], dataloaders_dict_tume[phase]):
                images = tubu[0].to(device)
                # images.shape→torch.Size([32, 3, 224, 224])
                labels = tubu[1].to(device)
                # labels.shape→torch.Size([32, 3])

                if args.triplet:
                    outputs, embed_vec = model(images)

                    # for embedding projector
                    embed_lab = labels[:,2]
                    #embedding projectorのためのリストを作成
                    embedvec_box_eval.append(embed_vec)
                    embedlab_box_eval.append(embed_lab)
                else:
                    outputs = model(images)
                
                labels = labels[:,0]
                
                
                index = len(labels) if len(labels) <= args.batch_size else args.batch_size  #batchサイズ調整用
                for i in range(index):
                    lab = float(labels[i]*(normalization[1]-normalization[0])+normalization[0])
                    out = float(outputs[i]*(normalization[1]-normalization[0])+normalization[0])

                    label.append(lab)
                    output.append(out)
                    outbox[f'{lab}'].append(out)


                    if lab-0.5 <= out <= lab+0.5:
                        acc_half[f'{lab}'] += 1
                        if lab-0.25 <= out <= lab+0.25:
                            acc_quarter[f'{lab}'] += 1
            
            if args.triplet:
                #catでテンソルを結合
                embedvec_box_eval = torch.cat(embedvec_box_eval, dim=0)
                embedvec_box_eval = embedvec_box_eval.cpu().detach().numpy()    #tensorをnumpyに変換
                embedvec_box_eval = embedvec_box_eval.tolist()                  #numpyをリストに変換

                embedlab_box_eval = torch.cat(embedlab_box_eval, dim=0)
                # len(embedvec_box_eval)→1081(validation粒数)
                # len(embedlab_box_eval)→1081(validation粒数)

                #vecs, metaを作成
                #vecs
                embedding_dir = os.path.join(f'{args.output_dir}/tsv')
                if not os.path.exists(embedding_dir):
                    os.makedirs(embedding_dir)           
                with open(f'{embedding_dir}/vecs.tsv', 'w') as fw:
                    csv_writer = csv.writer(fw, delimiter='\t')
                    csv_writer.writerows(embedvec_box_eval)
                #meta
                out_m=io.open(f'{embedding_dir}/meta.tsv', 'w', encoding='utf-8')

                [out_m.write(str(x)+'\n') for x in embedlab_box_eval]

                out_m.close()

        with open(f'{args.output_dir}/result_{phase}.txt', mode='a') as f: #末尾に追加モード
            f.write(f'\n【{phase}_data】\n'+f'r:{np.corrcoef(label, output)[0, 1]:.3f}\n')  #np.corrcoef()：与えられたデータの相関係数を計算
            print(f'\n【{phase}_data】\n'+f'r:{np.corrcoef(label, output)[0, 1]:.3f}')
            for i,o in enumerate(outbox):   #oには、キー値(1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)が入る
                mean.append(np.mean(outbox[o]))         #ラベルごとの平均が入ったリストができる
                median.append(np.median(outbox[o]))     #ラベルごとの中央値が入ったリストができる
                sd.append(np.std(outbox[o], ddof=1))    #ラベルごとの不偏標準偏差が入ったリストができる。ddof=1で不偏標準偏差(n-1で割るやつ)になる
                try:
                    acch = acc_half[o]*100/len(outbox[o])
                    accq = acc_quarter[o]*100/len(outbox[o])
                except ZeroDivisionError:
                    acch, accq = 0, 0
                acc_h.append(acch)
                acc_q.append(accq)
                f.write(f'Label:{o}, Mean:{mean[i]:.2f}, Median:{median[i]:.2f}, Sd:{sd[i]:.2f}, Acc(±0.5):{acc_h[i]:4.1f}%, Acc(±0.25):{acc_q[i]:4.1f}%\n')
                print(f'Label:{o}, Mean:{mean[i]:.2f}, Median:{median[i]:.2f}, Sd:{sd[i]:.2f}, Acc(±0.5):{acc_h[i]:4.1f}%, Acc(±0.25):{acc_q[i]:4.1f}%')
                plt.text(float(o)+0.1, median[i], f'←{median[i]:.2f}', backgroundcolor='white')
            f.write(f'-'*80+f'\nTotal Average                      Sd:{np.mean(sd):.2f}, Acc(±0.5):{np.mean(acc_h):4.1f}%, Acc(±0.25):{np.mean(acc_q):4.1f}%\n')
            print(f'-'*80+f'\nTotal Average                      Sd:{np.mean(sd):.2f}, Acc(±0.5):{np.mean(acc_h):4.1f}%, Acc(±0.25):{np.mean(acc_q):4.1f}%\n')


        #Excelファイル出力
        # mode = 'w' if phase == 'valid' else 'a'
        # withステートメントを使用することで、Excelファイルの書き込みが終了した時点で自動的にファイルを閉じることができる。
        with pd.ExcelWriter(f'{args.output_dir}/result_{phase}.xlsx', mode="w") as writer:  #pd.ExcelWriter()関数で、Excelファイルを作成するためのオブジェクトを作成。mode="w"は、書き込みモード。
            df = pd.DataFrame([mean, median, sd, acc_h, acc_q], index=['Mean', 'Median', 'Sd', 'Acc(±0.5)', 'Acc(±0.25)'], columns=number)  #pd.DataFrame()関数で、データを含むPandasのデータフレームを作成。ここでは、mean, median, sd, acc_h, acc_qという5つのリストを行に持ち、numberというリストを列に持つデータフレームを作成。indexパラメータは行のラベルを指定し、columnsパラメータは列のラベルを指定。
            df.to_excel(writer, sheet_name=f'{phase}')  #df.to_excel()関数を使用して、作成したデータフレームをExcelファイルに書き込む。sheet_nameパラメータは、シート名を指定し、writerオブジェクトは、作成したExcelファイルに書き込むためのもの。

        #分布図描画
        plt.title(f'color distribution ({phase}_data)')
        plt.xlabel('label')
        plt.ylabel('estimated value')
        plt.grid()
        plt.scatter(label, output, alpha=0.2, c='Green')    #plt.scatter()関数で、x軸にlabel、y軸にoutputを指定して、それらの値を散布図としてプロット。alpha=0.2で、散布図の透明度を指定(0に近づくほど透明になり、1に近づくほど不透明になる)。c='Green'は、散布図の色を指定。
        ymin, ymax = plt.ylim() #plt.ylim()関数で、現在のグラフのy軸の範囲(最小値と最大値)を取得。
        ymin, ymax = round(ymin), round(ymax)+0.5
        plt.yticks(np.arange(ymin, ymax, step=0.5)) #plt.yticks()関数で、y軸の目盛りを設定。np.arange(ymin, ymax, step=0.5)で、y軸の範囲をyminからymaxまで、0.5の間隔で区切った値を配列として取得。この配列をplt.yticks()関数の引数に渡すことで、y軸の目盛りを設定。
        plt.text(1.5, ymax-1.0, f'r = {np.corrcoef(label, output)[0, 1]:.3f}', size=12, backgroundcolor='white')    #plt.text()関数で、グラフにテキストを追加。ここでは、(1.5, ymax-1.0)の位置に、相関係数を表示するテキストを追加。「size=12」でテキストのサイズを指定。「backgroundcolor='white'」で背景色を白色に指定。
        plt.savefig(f'{args.output_dir}/scatter_{phase}.jpg')
        plt.clf()   #現在のグラフをクリアする。


        #正規分布＋度数分布（色ごと）
        pltc = ['b', 'k', 'r', 'g', 'y', 'c', 'm']
        os.makedirs(args.output_dir+f'/{phase}_dis', exist_ok=True)
        for c, n in enumerate(number):  #number = ['1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5']
            plt.title(f'{n} distribution ({phase}_data :{len(outbox[n])})')
            plt.xlabel('label')
            plt.ylabel('number of data')
            plt.hist(outbox[n], bins=40, alpha=0.2, color=pltc[c], label=n, range=(1.0, 5.0))   #plt.hist()は、ヒストグラムを描画するための関数。outbox[n]という配列から40本の棒グラフを描画し、それらの色をpltc[c]に設定。また、range=(1.0, 5.0)を指定しているため、x軸の範囲は1.0から5.0。
            plt.axvline(np.median(outbox[n]), color=pltc[c], linestyle='dashed')    #plt.axvline()は、指定したx座標に垂直な直線を描画するための関数。ここでは、np.median(outbox[n])の位置に、破線(linestyle='dashed')の直線を描画。
            ymax, ymin = plt.ylim()
            plt.text(np.median(outbox[n])+0.1, ymin*0.9, f'Median : {np.median(outbox[n]):.2f}', size ='large') #sizeに文字列'large'を指定すると、テキストのフォントサイズがデフォルトよりも大きくなる。
            plt.text(np.median(outbox[n])+0.1, ymin*0.85, f'σ : {np.std(outbox[n], ddof=1):.2f}', size ='large')
            pdf = norm.pdf(np.arange(1.0, 5.0, 0.1), np.mean(outbox[n]), np.std(outbox[n], ddof=1))*0.1*len(outbox[n])  #norm.pdf(x, loc, scale)は、平均loc、標準偏差scaleの正規分布における、確率密度関数の値を返す関数。np.arange(1.0, 5.0, 0.1)：1.0から4.9まで0.1刻みで値を生成(これがx軸の値の範囲)。pdfを計算する式は、norm.pdf()の結果に0.1（np.arange()で指定した0.1刻みの値）と、len(outbox[n])（outbox[n]の要素数）をかけたものです。これは、単位面積あたりの確率密度を求めるための補正係数を計算している。
            plt.plot(np.arange(1.0, 5.0, 0.1), pdf, color=pltc[c])
            plt.minorticks_on() #補助目盛線を有効にする。
            plt.grid(which='major', color='black', alpha=0.3)   #plt.grid()は、グリッド線を表示するための関数。引数whichには、メジャーグリッド線か、マイナーグリッド線かを指定。colorでグリッド線の色、alphaでグリッド線の透明度を指定。また、マイナーグリッド線の場合はlinestyleにマイナーグリッド線のスタイルを指定。
            plt.grid(which='minor', color='gray', linestyle=':', alpha=0.2) #linestyle=':'で、点線。
            plt.savefig(f'{args.output_dir}/{phase}_dis/{phase}_{n}.jpg')
            plt.clf()

        #度数分布（全色）
        datalen = 0
        for c, n in enumerate(number):
            plt.hist(outbox[n], bins=40, alpha=0.2, color=pltc[c], label=n, range=(1.0, 5.0))
            plt.axvline(np.median(outbox[n]), color=pltc[c], linestyle='dashed', linewidth=1)   #linewidthで、線の太さを指定
            ymax, ymin = plt.ylim()
            plt.text(np.median(outbox[n])+0.1, ymin*0.9, f'{np.median(outbox[n]):.2f}', color=pltc[c])
            datalen += len(outbox[n])

        plt.title(f'Frequency distribution ({phase}_data :{datalen})')  #Frequency distribution：度数分布
        plt.xlabel('label')
        plt.ylabel('number of data')
        plt.minorticks_on()
        plt.grid(which='major', color='black', alpha=0.3)
        plt.grid(which='minor', color='gray', linestyle=':', alpha=0.2)
        plt.savefig(f'{args.output_dir}/{phase}_dis/fre_dis.jpg')
        plt.clf()


        #正規分布（全色）
        for c, n in enumerate(number):
            plt.axvline(np.median(outbox[n]), color=pltc[c], linestyle='dashed', linewidth=1)
            ymax, ymin = plt.ylim()
            plt.text(np.median(outbox[n])+0.1, ymin*0.9, f'{np.std(outbox[n], ddof=1):.2f}', color=pltc[c])
            pdf = norm.pdf(np.arange(1.0, 5.0, 0.1), np.mean(outbox[n]), np.std(outbox[n], ddof=1))*0.1*len(outbox[n])
            plt.plot(np.arange(1.0, 5.0, 0.1), pdf, color=pltc[c])

        plt.title(f'Normal distribution ({phase}_data)')
        plt.xlabel('label')
        plt.ylabel('number of data')
        plt.grid()
        plt.savefig(f'{args.output_dir}/{phase}_dis/nor_dis.jpg')
        plt.clf()


    if "efficientnet" in args.feat_type:
        model.feat_ext.feat_ext.set_swish(memory_efficient=False)   #model.feat_ext.feat_ext.set_swish(memory_efficient=False)で、modelというオブジェクトの属性feat_extの中の属性feat_extにあるset_swish()メソッドを呼び出している。set_swish()メソッドは、ニューラルネットワークの活性化関数であるSwishを設定するために使用。memory_efficient引数は、Swish関数を計算するためのメモリの使用量を制御するために使用。引数のデフォルト値はFalseで、より高速なSwish計算を実現するためにより多くのメモリを使用。引数をTrueに設定すると、より少ないメモリでSwish関数を計算できるが、演算速度は遅くなる。
        torch.onnx.export(model, torch.rand(1,3,224,224).to(device),f"{args.output_dir}/{args.feat_type}_{args.type}.onnx",     #学習モデルonnxファイルの出力部分
            verbose=False, opset_version=12, input_names=['input'],
            output_names=['color'],
            dynamic_axes={
                        'input':{0:"batch_size",2:"height",3:"width"},
                        'color':{0:"batch_size",1:"color_value"},
            },
        )
        #torch.onnx.export()は、PyTorchモデルをONNX形式にエクスポートするための関数
        #引数
        #model: ONNXにエクスポートするPyTorchモデル
        #args: 入力テンソル
        #エクスポートされたONNXモデルのファイルパス
        #verbose: ログ出力の有効化/無効化を制御するフラグ
        #opset_version: 使用するONNXオペレーションのバージョンを指定
        #input_names: ONNXモデルの入力名
        #output_names: ONNXモデルの出力名
        #dynamic_axes: ONNXモデルにおいて、どの軸が可変であるかを指定。ここでは、inputのバッチサイズ、高さ、幅が可変であり、colorのバッチサイズ、カラーチャンネル数が可変であることを指定

    if args.save_half:
        model.half()
        torch.onnx.export(model.half(), torch.rand(1,3,224,224).to(device).half(),f"{args.output_dir}/{args.feat_type}_{args.type}_fp16.onnx", 
            verbose=False, opset_version=12, input_names=['input'],
            output_names=['color'],
            dynamic_axes={
                        'input':{0:"batch_size",2:"height",3:"width"},
                        'color':{0:"batch_size",1:"color_value"},
            },
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, default='CU 117 Test', help='project_name for wandb')    #wandbのproject名
    parser.add_argument('--wandb_entity', type=str, default='r_shimazu31', help='entity for wandb')    #wandbのproject名
    parser.add_argument('-outbase','--output_basedir', type=str, default='output_cu117', help='output_dir') #出力のベースディレクトリ
    parser.add_argument('-optimizer','--optimizer', type=str, default="Adam", help='choose the image transform class whether,adam|adamW|Lamb|RMSProp|SGD')  #optimizerの選択
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('-b','--batch_size', type=int, default=32, help='batch size')
    # parser.add_argument('-e','--epochs', type=int, default=300, help='epochs')
    parser.add_argument('-e','--epochs', type=int, default=1, help='epochs')
    # parser.add_argument('-img_size','--img_size', type=int, default=400, help='img size')
    parser.add_argument('-img_size','--img_size', type=int, default=224, help='img size not crop')  #画像サイズ   
    parser.add_argument('-lr','--lr', type=float, default=1e-4, help='learning_rate')  #学習率
    parser.add_argument('--aug', type=int, default=1, help='whether to use augmentation')   #arg=8だったら、180度回転した画像も作成
    parser.add_argument('-nl','--n_layers', type=int, default=1, help='Number of layers in classifications layers')
    parser.add_argument('-u','--n_units', type=int, default=1, help='number of units in classification layers')
    parser.add_argument('-feat_type','--feat_type', type=str, default="vit_small_patch16_224", help='type of feature extractor')  #特徴量抽出器の種類
    parser.add_argument('-type','--type', type=str, default="Transformer", help='type of classification block Linear|LSTM|Transformer')  #分類ブロックの種類
    # parser.add_argument('-image_type','--image_type', type=str, default="LGB", help='choose the image transform class whether, RGB|LGB|RGL|RLB|VGB|RVB|RGV')    #色空間の選択
    parser.add_argument('-image_type','--image_type', type=str, default="RGB", help='choose the image transform class whether, RGB|LGB|RGL|RLB|VGB|RVB|RGV')    #色空間の選択
    parser.add_argument('--normalization', nargs='+', type=float, default=[1.5, 4.5], help='data normalization')  # height, width
    parser.add_argument('--use_aug', default=False, action="store_true", help='Whether to use data augmentation')   #データ拡張をするかどうか
    parser.add_argument("-l2",'--l2_norm_embed', type=ast.literal_eval, default=True, help='Whether to use l2 normalized embedding')  # height, width
    parser.add_argument('--save_half', default=True, action="store_true", help='Whether to save fp16 model in onnx')  # onnxでfp16モデルを保存するかどうか
    
    parser.add_argument('--triplet', type=ast.literal_eval, default=False, help='Whether to train with triplet learning')   #距離学習ありかなしか
    parser.add_argument('--tume', type=ast.literal_eval, default=False, help='Whether to train with tume learning')   #爪学習ありかなしか
    parser.add_argument('-c_rate','--c_rate', type=float, default=1, help='cost_rate')    #色推定Lossの比率指定
    parser.add_argument('-tume_rate','--tume_rate', type=float, default=0, help='tume_rate')  #爪学習Lossの比率指定
    parser.add_argument('-t_rate', '--t_rate', type=float, default=0, help='triplet_loss_rate')  #距離学習Lossの比率指定

    parser.add_argument('-const','--const', type=float, default=0.2, help='light and contrast param')   #コントラスト調整幅指定
    parser.add_argument('-bright','--bright', type=float, default=0.2, help='light and contrast param') #torchvisionデータの明るさ
    parser.add_argument('-train_count','--train_count', type=int, default=2, help='train count')
    parser.add_argument('-balance_rate','--balance_rate', type=float, default=1.5, help='balance_rate') #データバランス補正用の許容限度倍率の指定

    parser.add_argument('-dir_name','--dir_name', type=str, default='single', help='dir name') #ディレクトリの名前
    
    parser.add_argument('-margin','--margin', type=float, default=0.2, help='margin')   #距離学習のmarginの指定
    args = parser.parse_args()  #使用される引数の解析

    seed=1234

    #Lossの比率指定
    if args.triplet and not args.tume:
        args.c_rate = 1-args.t_rate
    elif not args.triplet and args.tume:
        args.c_rate = 1-args.tume_rate
    
    #dir_nameの指定
    if args.triplet and not args.tume:
        args.dir_name = 'triplet'
    elif not args.triplet and args.tume:
        args.dir_name = 'tume'
    elif args.triplet and args.tume:
        args.dir_name = 'mtl'
    else:
        args.dir_name = 'single'
    
    #出力ディレクトリを指定
    if args.triplet:
        args.output_dir = f'{args.output_basedir}/{args.dir_name}/{args.feat_type}_{args.optimizer}_{args.lr}_{args.image_type}_cost_weight{args.c_rate}-triplet_weight{args.t_rate}-margin{args.margin}-tume_weight{args.tume_rate}_torch-const{args.const}-bright{args.bright}_balance{args.balance_rate}_seed{seed}'   #出力ディレクトリを指定
    else:
        args.output_dir = f'{args.output_basedir}/{args.dir_name}/{args.feat_type}_{args.optimizer}_{args.lr}_{args.image_type}_cost_weight{args.c_rate}-triplet_weight{args.t_rate}-tume_weight{args.tume_rate}_torch-const{args.const}-bright{args.bright}_balance{args.balance_rate}_seed{seed}'   #出力ディレクトリを指定
    os.makedirs(args.output_dir, exist_ok=True)

    torch_fix_seed(seed)    #seed値固定関数
    main(args)