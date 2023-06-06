import torch
import numpy as np
from utils.utils import *
from al_strategies import ralif_query
from extract_feat import extract_features_func
from tqdm import tqdm
import torch.nn as nn
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoProcessor, CLIPVisionModel

import torch.multiprocessing as mp
import scipy.stats as stats
import resnet

import argparse
from dataset_config import datasets_transforms
import custom_inaturalist
from torch.utils.data import Subset

if __name__ == "__main__":
  ddp_setup()
  torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', help='model: pre-ResNet18, scratch-ResNet18, clip', type=str, default='pre-ResNet18')
  parser.add_argument('--dataset', help='dataset: cifar10, cifar100, iNaturalist', type=str, default='cifar10')
  parser.add_argument('--tau', help='threshold for truncated IL', type=int, default=4)
  parser.add_argument('--init_size', help='initial size for AL', type=int, default=1000)
  parser.add_argument('--budget_size', help='budget size for AL', type=int, default=1000)
  parser.add_argument('--cycles', help='AL cycles', type=int, default=10)
  parser.add_argument('--res_ck', help='the path for saving results', type=str, default='./results_ck/')
 
  args = parser.parse_args()
  if not os.path.exists(args.res_ck):
    os.makedirs(args.res_ck)
  if not os.path.exists(args.res_ck + 'features'):
    if dist.get_rank() == 0:
      os.makedirs(args.res_ck + 'features')
      os.makedirs(args.res_ck + 'labels')
      os.makedirs(args.res_ck + 'models')
      os.makedirs(args.res_ck + 'results') 
  dist.barrier()

  if not os.path.exists(args.res_ck + 'init_info.npy'):
    if dist.get_rank() == 0: 
      if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        init_info = init_set(args)
      else:
        init_info = inaturalist_init_set(args)

  dist.barrier()
  init_info = np.load(args.res_ck + 'init_info.npy', allow_pickle=True).item()
  if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    al_set, begin_set, rest_set = init_info['al_set'], init_info['begin_set'], init_info['rest_set'] 
  else:
    train_indices, begin_set, rest_set, test_indices = init_info['train_indices'], init_info['begin_set'], init_info['rest_set'], init_info['test_indices']

  select_pool = rest_set.copy()
  assert set(begin_set) & set(select_pool) == set()
  for iteration in range(args.cycles):
    if os.path.exists(args.res_ck + 'results/select_pool_%d.npy' % iteration):
      continue
    else:
      if os.path.exists(args.res_ck + 'results/begin_set_%d.npy' % (iteration - 1)):
        begin_set = np.load(args.res_ck + 'results/begin_set_%d.npy' % (iteration - 1), allow_pickle=True)
      if os.path.exists(args.res_ck + 'results/select_pool_%d.npy' % (iteration - 1)):
        select_pool = np.load(args.res_ck + 'results/select_pool_%d.npy' % (iteration - 1), allow_pickle=True)
      else:
        select_pool = np.array([i for i in select_pool if i not in begin_set])
      assert set(begin_set) & set(select_pool) == set()
      
      ### estract features
      if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10('/home/datasets/cifar10/', train=True, download=False, transform = datasets_transforms[args.model][args.dataset]['test'])
        test_dataset = torchvision.datasets.CIFAR10('/home/datasets/cifar10/', train=False, download=False, transform = datasets_transforms[args.model][args.dataset]['test'])
        label_dataset = torchvision.datasets.CIFAR10('/home/datasets/cifar10/', train=True, download=False, transform = datasets_transforms[args.model][args.dataset]['train'])
        args.cls_num = 10
      elif args.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100('/home/datasets/', train=True, download=False, transform = datasets_transforms[args.model][args.dataset]['test'])
        test_dataset = torchvision.datasets.CIFAR100('/home/datasets/', train=False, download=False, transform = datasets_transforms[args.model][args.dataset]['test'])
        label_dataset = torchvision.datasets.CIFAR100('/home/datasets/', train=True, download=False, transform = datasets_transforms[args.model][args.dataset]['train'])
        args.cls_num = 100
      elif args.dataset == 'iNaturalist':
        inaturalist_feat_dataset = custom_inaturalist.INaturalist(root='/home/datasets/inaturalist/', version='2018', target_type='super', transform = datasets_transforms[args.model][args.dataset]['test'])
        train_dataset = Subset(inaturalist_feat_dataset, train_indices) 
        test_dataset = Subset(inaturalist_feat_dataset, test_indices) 
        inaturalist_label_dataset = custom_inaturalist.INaturalist(root='/home/datasets/inaturalist/', version='2018', target_type='super', transform = datasets_transforms[args.model][args.dataset]['train'])
        label_dataset = Subset(inaturalist_label_dataset, train_indices)
        args.cls_num = 14

      if args.model == 'clip':
        if os.path.exists(args.res_ck + 'features/train_features_0.npy'):
          total_train_feats = np.load(args.res_ck + 'features/train_features_0.npy')
          total_train_labels = np.load(args.res_ck + 'labels/train_labels_0.npy')
          args.feat_dim = total_train_feats.shape[1]
        else:
          total_train_feats, total_train_labels = extract_features_func(args, label_dataset, train_dataset, test_dataset, begin_set, iteration)
      else:
        if os.path.exists(args.res_ck + 'features/train_features_%d.npy' % iteration):
          total_train_feats = np.load(args.res_ck + 'features/train_features_%d.npy' % iteration)
          total_train_labels = np.load(args.res_ck + 'labels/train_labels_%d.npy' % iteration)
          args.feat_dim = total_train_feats.shape[1]
        else:
          total_train_feats, total_train_labels = extract_features_func(args, label_dataset, train_dataset, test_dataset, begin_set, iteration)
     
      new_select = ralif_query(args, begin_set, total_train_feats, total_train_labels, iteration, select_pool)

      begin_set = np.append(begin_set, new_select)
      select_pool = np.array([i for i in select_pool if i not in begin_set])
      
      if dist.get_rank() == 0:
        np.save(args.res_ck + 'results/select_%d.npy' % iteration, new_select)
        np.save(args.res_ck + 'results/begin_set_%d.npy' % iteration, begin_set)
        np.save(args.res_ck + 'results/select_pool_%d.npy' % iteration, select_pool)

      assert set(begin_set) & set(select_pool) == set()
  destroy_process_group()
