import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet#, resnet50, resnet101, resnet152, resnet18, resnet34

import itertools
from time import time
from tqdm import tqdm
import random

import os
from PIL import Image
from torchvision import transforms

import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score #,f1_score, roc_auc_score #, precision_score, recall_score, confusion_matrix
from itertools import product
from sklearn.utils import shuffle


def generate_nn(n, blocks_type, nf=64):

  for ni in n: 
    if ni < 1 or blocks_type not in [0, 1] or nf < 1:
      print("Все n[i], nf должны быть >= 1, blocks_type in list(0, 1)")
      return False

  result = OrderedDict([
    ('conv1', nn.Conv2d(3, nf, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)),
    ('bn1', nn.BatchNorm2d(nf, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
    ('relu', nn.ReLU(inplace=True)),
    ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)),
  ])

  for i, ni in enumerate(n):
    # if blocks_type == 1:
    #   i += 1
    sequential_i = [] 
    for j in range(ni):
      # print(i,j)

      if blocks_type == 0:
        k = 0
        s = 1
        if i == 0:
          downsample = None
        else:
          if j == 0:
            if i >= 1:
              s = 2
            k = -1
            downsample = nn.Sequential(
              nn.Conv2d(nf * 2**(i + k), nf * 2**i, kernel_size=(1, 1), stride=(2, 2), bias=False),
              nn.BatchNorm2d(nf * 2**i, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            )      
          else:
            downsample = None

        sequential_i.append(
          resnet.BasicBlock(
            nf * 2**(i + k),
            nf * 2**i, 
            downsample=downsample,
            stride = s,          
          )
        )

      elif blocks_type == 1:
        k = 1
        s = 1
        if j == 0:
          if i >= 1:
            k = 2
            s = 2
          k1 = 4
          downsample = nn.Sequential(
            nn.Conv2d(nf * 2**i * k, nf * 2**i * k1, kernel_size=(1, 1), stride=s, bias=False),
            nn.BatchNorm2d(nf * 2**i * k1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          )

        else:
          k = 4
          downsample = None

        sequential_i.append(
          resnet.Bottleneck(
            nf * 2**i * k,
            nf * 2**i, 
            downsample=downsample,
            stride = s,
          )
        )
        
    result[f'layer{i + 1}'] = nn.Sequential(*sequential_i)

  result['avgpool'] = nn.AdaptiveAvgPool2d(output_size=(1, 1))
  k = 1
  if blocks_type == 1:
    k = 4
  # result['fc'] = nn.Linear(in_features=512 * k, out_features=1000, bias=True)

  return nn.Sequential(result)


def prepare_input():

  os.system("wget http://m.w3m.ru/rt/p1/frames.zip")
  os.system("unzip frames.zip -d frames")

  def f1(folder):
    r = []
    for f in os.listdir(folder):
      path = os.path.join(folder, f)
      if os.path.isfile(path):
        r.append(path)
      else:
        r += f1(path)
    return r

  return f1('frames')


def generate_dataset(images_paths, samples_count=int(10**(1/2)), len_=1000):
  rs = 38
  random.seed(rs)
  img_idxs = random.sample(range(len(images_paths)), samples_count)
  imgs = [Image.open(images_paths[idx]) for idx in img_idxs]

  convert_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
  ])

  nf_list = [1, 2] + [int(x**1.7) for x in range(2, 23)]

  k = 2
  ns_list = \
    list(itertools.product(range(1, 7 * k, 1), range(1, 9 * k, 1), range(1, 13 * k, 2), range(1, 7 * k, 1), [0], nf_list)) \
    + \
    list(itertools.product(range(1, 7 * k, 1), range(2 - 1, 16 * k, 2), range(3 - 2, 72 * int(k * 2 / 2), 3), range(1, 7 * k, 1), [1], nf_list))    

  random.seed()
  random.shuffle(ns_list)
  
  i1 = 0
  res = []
  for i in tqdm(ns_list):
    print(i)
    i1 += 1
    nn_ = generate_nn(list(i)[:4], list(i)[4], list(i)[5])

    _res = []
    for img in imgs:
      time1 = time()
      nn_(
        convert_tensor(img).unsqueeze(0)
      )
      time2 = time()
      _res.append(time2 - time1)
    print(_res)
    res.append([*i, sum(_res) / len(_res)])

    if i1 >= len_:
      break

  return res


def split(df, tc, rs, ts, vs): # dataframe, target column, random state, test size, valid size
    f_all, t_all = df.drop(tc, axis=1), df[tc] # features, target (all)
    f_tmp, f_test, t_tmp, t_test = train_test_split(
        f_all, t_all, test_size=ts, random_state=rs#, stratify=t_all
    ) 
    f_train, f_valid, t_train, t_valid = train_test_split(
        f_tmp, t_tmp, test_size=vs, random_state=rs#, stratify=t_tmp
    )
    print(f_train.shape, f_valid.shape, f_test.shape, t_train.shape, t_valid.shape, t_test.shape)
    return f_train, f_valid, f_test, t_train, t_valid, t_test
  
  
def search_model(models, metrics, data, round_=4): #show trees
    f_train, f_valid, t_train, t_valid = data
    r = [] #result
    for md in models: 
        for i, p_a in tqdm(enumerate(
            list(product( #params array
                *md['params'].values()
            ))
        )):
            p_d = dict(zip(md['params'].keys(), p_a)) #params dict
            model = md['model'](**p_d)
            model.fit(f_train, t_train)
            as_ = []
            for metric in metrics:
                as_.append(
                    round(
                        metric(
                            t_valid,
                            model.predict(f_valid),
                        ),
                        round_,
                    )
                )
            r.append([
                as_,
                model, 
                p_d
            ])
    return r
  

def check_model(df, model, metrics, n): # 
    r = [[] for i in range(len(metrics))]
    print(r)
    for i in range(n):
        f_train, f_valid, f_test, t_train, t_valid, t_test = split(df, tc, random.randint(0,100), ts, 1e-5)
        model.fit(f_train, t_train)
        if isinstance(model, lrc):
          print(model.coef_)
        t_pred = model.predict(f_test)
        for j, metric in enumerate(metrics):
            ri = round(
              metric(t_test, t_pred),
              4,
            ),
            r[j].append(ri[0])
            print(ri[0], end=' ')
        print(sum(t_test) / len(t_test))

    return r
