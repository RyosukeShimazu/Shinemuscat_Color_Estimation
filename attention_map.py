import argparse
import glob
import logging
import os
import random
import time
# from io import BytesIO
import faiss # original add (2021/12/22)

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
from tensorboardX import SummaryWriter # 2022/11/02 追加
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
        image_path = self.data[index]
        x = cv2.imread(image_path)           #画像の読み込み
        resized = cv2.resize(x.copy(), dsize=(self.H, self.W))                                      # 224x224にリサイズ
        x = image_converter(resized, type=self.image_type)                                          # 画像の色空間を変換
        x = Image.fromarray(x)      #NumPy配列xをPIL(Python Imaging Library)のImageオブジェクトに変換
        x = self.transform(x)                                                        # Albumentationsでの変換はNumPy配列を引数にとる。

        y_c = self.target_c[index]  # 正解ラベル

        return x, y_c, image_path, resized


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

#Attention Weightを取得するための関数
def extract(pre_model, target, inputs):
    feature = None
    def forward_hook(module, inputs, outputs):              #forward_hookという関数を定義。この関数は特定のモジュールの順伝播時に実行される関数で、この関数内で順伝播の出力を「blocks」に保存。
        #順伝搬の出力をfeaturesというグローバル変数に記録する
        global blocks                                       #グローバル変数「blocks」を定義。この変数は後にモジュールの順伝播時の出力を格納するために使用。
        blocks = outputs.detach()
    #コールバック関数を登録する
    handle = target.register_forward_hook(forward_hook)     #targetと指定されたモジュールが順伝播を行うたびに、forward_hook関数が自動的に呼び出されるようになる。
    #推論する
    pre_model.eval()
    pre_model(inputs)
    #コールバック関数を解除する
    handle.remove()
    return blocks

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'GPU:{torch.cuda.is_available()}, device:{device}\n')            #「torch.cuda.is_available()」：現在の環境でCUDAが利用可能である場合にTrue、利用できない場合はFalseを返す。

    arguments = vars(args)                                          #vars()関数は、辞書型に変換
    print(f'arguments：{arguments}\n')

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

    #Attention Rolloutの可視化
    image_dir = args.output_dir
    #学習済みモデルを読み込む
    model.load_state_dict(torch.load('/misc/Work32/r_shimazu31/train_data_clean_resize224/output/single/vit_small_patch16_224_Adam_0.0001_RGB_cost_weight1-triplet_weight0-tume_weight0_torch-const0.2-bright0.2_balance1.5_seed1234/model_best_valid.pth'))
    model.eval()

    image_path_list = []
    output_list = []
    label_list = []
    abs_output_label_list = []
    for i, batch in enumerate(dataloaders_dict['valid']):
        tubus = batch[0]
        labels = batch[1]
        image_paths = batch[2]
        tubu_images = batch[3]
        batch_size = tubus.shape[0]
        for j in range(batch_size):
            tubu = tubus[j]
            label = labels[j]
            image_path = image_paths[j]
            harvest_day = image_path.split('/')[-3]
            color_name = image_path.split('/')[-2]
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            tubu_image = tubu_images[j]
            image_path_list.append(image_path)
            
            tubu = tubu.unsqueeze(0).to(device)
            output = model(tubu)
            # output.shape torch.Size([1, 1])
            lab = float(label[0]*(normalization[1]-normalization[0])+normalization[0])
            out = float(output*(normalization[1]-normalization[0])+normalization[0])
            label_list.append(lab)
            output_list.append(out)
            
            # 絶対値を計算
            abs_output_label = abs(out - lab)
            abs_output_label_list.append(abs_output_label)

            # Attention Mapの可視化
            attentions = []
            for block in range(len(model.feat_ext.feat_ext_fn.blocks)):
                target_module = model.feat_ext.feat_ext_fn.blocks[block].attn.attn_drop#attn_drop層を取り出す
                attention = extract(model.feat_ext.feat_ext_fn, target_module, tubu)#Attention Weightを取得
                attentions.append(attention)
            # ViTのパッチの数を取得
            num_patches = model.feat_ext.feat_ext_fn.patch_embed.num_patches
            # ViTのレイヤーの数を取得
            num_layers = len(attentions)
            fig, axs = plt.subplots(1, num_layers, figsize=(20, 20))
            # 各レイヤーでのattention weight
            for layer_idx in range(num_layers):
                attention = attentions[layer_idx][0]  # 最初のバッチのアテンションマップを取得．torch.Size([6(ヘッド数), 197(パッチ数+クラストークン), 197(パッチ数+クラストークン)])
                cls_attention = attention[:, 0, 1:].detach()#クラストークンが画像内の各パッチにどの程度注意を払っているかを取得．torch.Size([6(ヘッド数), 196(パッチ数)])6つのアテンションヘッドそれぞれに対して、クラストークンから196個のパッチへのアテンションの重みを表す
                patch_attention = cls_attention.mean(0)  # head方向の平均．torch.Size([196])
                patch_attention_map = patch_attention.view(int(num_patches ** 0.5), int(num_patches ** 0.5))#torch.Size([14, 14])
                axs[layer_idx].imshow(tubu_image)
                axs[layer_idx].imshow(patch_attention_map.cpu(), cmap='jet', alpha=0.6) # 勝手にpatch_attention_mapは224*224にリサイズされる
                # im = axs[layer_idx].imshow(patch_attention_map.cpu(), cmap='jet', alpha=0.6)# 勝手にpatch_attention_mapは224*224にリサイズされる
                # im = axs[layer_idx].imshow(patch_attention_map.cpu(), cmap='hot', alpha=0.6)# 勝手にpatch_attention_mapは224*224にリサイズされる
                axs[layer_idx].set_title(f'Layer {layer_idx+1}')
                axs[layer_idx].axis('off')
            plt.tight_layout()
            attention_map_save_dir = os.path.join(image_dir, 'attention_map', harvest_day, color_name, image_name)
            if not os.path.exists(attention_map_save_dir):
                os.makedirs(attention_map_save_dir)
            fig.savefig(f'{attention_map_save_dir}/layer_attention_map.png')  # 各レイヤーのアテンションマップを保存
            plt.close(fig)  # 現在の図を閉じる
            # ↑のレイヤー全ての平均
            avg_attention = torch.mean(torch.stack([attentions[layer_idx][0][:, 0, 1:].detach() for layer_idx in range(num_layers)]), dim=0)#torch.Size([6, 196])
            avg_patch_attention = avg_attention.mean(0)#torch.Size([196])
            avg_patch_attention_map = avg_patch_attention.view(int(num_patches ** 0.5), int(num_patches ** 0.5))#torch.Size([14, 14])
            plt.figure()  # 新しいfigureを作成
            plt.title('Average Patch Attention')
            im_avg = plt.imshow(avg_patch_attention_map.cpu(), cmap='jet')  # 戻り値をim_avgに格納
            # im_avg = plt.imshow(avg_patch_attention_map.cpu(), cmap='hot')  # 戻り値をim_avgに格納
            plt.colorbar(im_avg)  # カラーバーを追加
            plt.axis('off')
            plt.savefig(f'{attention_map_save_dir}/average_attention_map.png')  # 平均アテンションマップを保存
            plt.close()
    # DataFrameを作成
    df = pd.DataFrame({
        'image_path': image_path_list,
        'label': label_list,
        'output': output_list,
        'abs_output_label': abs_output_label_list
    })
    # 既存のDataFrameをソート
    df_sorted = df.sort_values(by='abs_output_label')
    # Excelファイルに出力
    excel_path = os.path.join(image_dir, 'output_data.xlsx')
    df_sorted.to_excel(excel_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-outbase','--output_basedir', type=str, default='20240217_attention_map', help='output_dir') #出力のベースディレクトリ
    parser.add_argument('-optimizer','--optimizer', type=str, default="Adam", help='choose the image transform class whether,adam|adamW|Lamb|RMSProp|SGD')  #optimizerの選択
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('-b','--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-e','--epochs', type=int, default=300, help='epochs')
    # parser.add_argument('-e','--epochs', type=int, default=2, help='epochs')
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

    args.output_dir = f'{args.output_basedir}'   #出力ディレクトリの指定
    os.makedirs(args.output_dir, exist_ok=True)

    torch_fix_seed(seed)    #seed値固定関数
    main(args)