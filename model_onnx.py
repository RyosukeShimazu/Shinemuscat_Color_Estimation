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

    model = ColorModel(arguments).to(device)    #arguments：vars(args)

    #モデル構造の可視化
    summary(model=model, input_size=(args.batch_size, 3, 224,224))
    #学習済みモデルを読み込む
    model.load_state_dict(torch.load('/home/usr22/r_shimazu31/work32/train_data_clean_resize224/output/single/vit_small_patch16_224_Adam_0.0001_RGB_cost_weight1-triplet_weight0-tume_weight0_torch-const0.2-bright0.2_balance1.5_seed1234/model_best_valid.pth'))
    model.eval()
    torch.onnx.export(model, torch.rand(1,3,224,224).to(device),"color_model_fp32.onnx", 
        verbose=False, opset_version=12, input_names=['input'],
        output_names=['color'],
        dynamic_axes={
                    'input':{0:"batch_size",2:"height",3:"width"},
                    'color':{0:"batch_size",1:"color_value"},
        },
    )


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

    torch_fix_seed(seed)    #seed値固定関数
    main(args)