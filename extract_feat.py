import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from utils.utils import *
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import resnet
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AutoProcessor, CLIPVisionModel

def extract_frozen_feat(model, dataloader):
  total_features = []
  total_labels = []

  with torch.no_grad():
    for i, (data, labels) in enumerate(tqdm(dataloader)):
      data = data.cuda(non_blocking=True)
      labels = labels.cuda(non_blocking=True)
      total_features.append(model(data))
      total_labels.append(labels)

  total_features = torch.cat(total_features, dim=0)
  total_labels = torch.cat(total_labels, dim=0)
  return total_features, total_labels

def clip_extract_frozen_feat(model, dataloader):
  with torch.no_grad():
    features, labels = [], []
    for data, label in tqdm(dataloader):
      outputs = model(data)
      pooled_output = outputs.pooler_output
      features.append(pooled_output)
      labels.append(label)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
  return features, labels


class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.
    
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def extract_features_func(args, label_dataset, train_dataset, test_dataset, begin_set, iteration):
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 300, shuffle=False, num_workers=4)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=300, shuffle=False, num_workers=4)
  label_dataloader = torch.utils.data.DataLoader(label_dataset, batch_size = 128, sampler=SubsetRandomSampler(begin_set))
   
  if dist.get_rank() == 0:
    print('AL Cycles', iteration, '---', 'Labeled set size', len(begin_set), '---', 'Train Begin')
  
  if args.model == 'clip':
    args.feat_dim = 768; args.epoch = 100; args.lr = 0.01
    train_feats = torch.zeros([len(train_dataset), args.feat_dim], device='cuda')
    train_labels = torch.zeros([len(train_dataset)], device='cuda').long()
    test_feats = torch.zeros([len(test_dataset), args.feat_dim], device='cuda')
    test_labels = torch.zeros([len(test_dataset)], device='cuda').long()
    model = CLIPVisionModel.from_pretrained("/mnt/huggingface_models/clip-vit-base-patch16")
  else:
    args.feat_dim = 512
    train_feats = torch.zeros([len(train_dataset), args.feat_dim], device='cuda')
    train_labels = torch.zeros([len(train_dataset)], device='cuda').long()
    test_feats = torch.zeros([len(test_dataset), args.feat_dim], device='cuda')
    test_labels = torch.zeros([len(test_dataset)], device='cuda').long()

    if args.model == 'pre-ResNet18':
      args.epoch = 200; args.lr = 0.01
      model = torchvision.models.resnet18(pretrained=True)
      model.fc = nn.Linear(in_features=args.feat_dim, out_features=args.cls_num)
      model.cuda()
      model = train_model(model, label_dataloader, args.epoch, args.lr, 1e-4, test_dataloader)
    elif args.model == 'scratch-ResNet18':
      args.epoch = 200; args.lr = 0.1
      model = resnet.ResNet18(args.cls_num).cuda()
      model = scratch_train_model(model, label_dataloader, args.epoch, args.lr, 1e-4, test_dataloader)
    else:
      print('No such model')

  if args.model == 'pre-ResNet18' or args.model == 'scratch-ResNet18':
    if dist.get_rank() == 0:
      print('AL Cycles', iteration, '---', 'Labeled set size', len(begin_set), '---', 'Train Finished: Val Acc: {:.3f}'.format(get_acc(model, test_dataloader) * 100), flush=True)
    if args.model == 'pre-ResNet18':
      torch.save(model.fc.state_dict(), args.res_ck + 'models/models_%d.pt' % (iteration))
      model.fc = nn.Identity()
    else:
      torch.save(model.linear.state_dict(), args.res_ck + 'models/models_%d.pt' % (iteration))
      model.linear = nn.Identity()
  
  if args.model == 'pre-ResNet18' or args.model == 'scratch-ResNet18':
    model.eval()
    train_feats, train_labels = extract_frozen_feat(model, train_dataloader)
    test_feats, test_labels = extract_frozen_feat(model, test_dataloader) 
  else:
    model.eval()
    train_feats, train_labels = clip_extract_frozen_feat(model, train_dataloader)
    test_feats, test_labels = clip_extract_frozen_feat(model, test_dataloader)

  if dist.get_rank() == 0:
    np.save(args.res_ck + 'features/train_features_%d.npy' % iteration, train_feats.cpu().numpy())
    np.save(args.res_ck + 'labels/train_labels_%d.npy' % iteration, train_labels.cpu().numpy())
    np.save(args.res_ck + 'features/test_features_%d.npy' % iteration, test_feats.cpu().numpy())
    np.save(args.res_ck + 'labels/test_labels_%d.npy' % iteration, test_labels.cpu().numpy())
  
  dist.barrier()

  dist.broadcast(train_labels, src=0)
  dist.broadcast(train_feats, src=0)
  dist.broadcast(test_labels, src=0)
  dist.broadcast(test_feats, src=0)
  return train_feats.cpu().numpy(), train_labels.cpu().numpy()
