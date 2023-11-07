import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader  
from PIL import Image
import os
import pickle
import torch.distributed as dist
from sklearn.metrics import pairwise_distances
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import scipy.stats as stats

def init_set(args):
    all_num = 50000
    al_set = np.arange(all_num)
    begin_set = np.random.choice(al_set, args.init_size, replace=False)
    rest_set = np.array(list(set(al_set) - set(begin_set)))
    init_info = {'begin_set':begin_set, 'rest_set': rest_set, 'al_set': al_set}
    np.save('./results_ck/init_info.npy', init_info)
    return init_info

def inaturalist_init_set(args):
    all_num = 461939
    train_num = int(all_num * 0.8)
    test_num = all_num - train_num
    train_indices = np.random.choice(np.arange(all_num), train_num, replace=False)
    test_indices = np.array(list(set(np.arange(all_num)) - set(train_indices)))
    assert len(test_indices) == test_num
    al_set = np.arange(train_num)
    begin_set = np.random.choice(al_set, args.init_size, replace=False)
    rest_set = np.array(list(set(al_set) - set(begin_set)))
    init_info = {'begin_set':begin_set, 'rest_set': rest_set, 'train_indices':train_indices, 'test_indices':test_indices}
    np.save('./results_ck/init_info.npy', init_info)
    return init_info

def train_model(model, train_dataloader, max_epoch, lr, weight_decay, test_dataloader):
  model.cuda()
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch * len(train_dataloader))
  criterion = nn.CrossEntropyLoss()

  for epoch in range(max_epoch):
    model.train()
    for i, (data, label) in enumerate(train_dataloader):
      data = data.cuda(non_blocking=True); label = label.cuda(non_blocking=True).long()
      logits = model(data)
      loss = criterion(logits, label)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step() 
      lr_scheduler.step()
    if dist.get_rank() == 0:
      if epoch % 20 == 0 or epoch == 199:
        print('Epoch', epoch, '---', 'Val Acc: {:.3f}'.format(get_acc(model, test_dataloader) * 100), flush=True)
  return model

def scratch_train_model(model, train_dataloader, max_epoch, lr, weight_decay, test_dataloader):
  model.cuda()
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
  lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])
  criterion = nn.CrossEntropyLoss()

  for epoch in range(max_epoch):
    model.train()
    for i, (data, label) in enumerate(train_dataloader):
      data = data.cuda(non_blocking=True); label = label.cuda(non_blocking=True).long()
      logits = model(data)
      loss = criterion(logits, label)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    lr_scheduler.step()
    if epoch % 20 == 0 or epoch == 199:
      print('Epoch', epoch, '---', 'LR', lr_scheduler.get_last_lr(), 'Val Acc: {:.3f}'.format(get_acc(model, test_dataloader) * 100), flush=True)
  return model

class CustomLogisticRegression(nn.Module):
  def __init__(self, input_dim, cls_num):
    super(CustomLogisticRegression, self).__init__()
    self.fc = nn.Linear(input_dim, cls_num, bias=True)

  def forward(self, x):
    x = self.fc(x)
    return x

def get_acc(model, dl):
  hit, tot = 0, 0
  model.eval()
  with torch.no_grad():
    for data, label in dl:
      data = data.cuda(non_blocking=True); label = label.cuda(non_blocking=True).long()
      logits = model(data)
      hit += (logits.topk(1, dim=1)[1] == label.view([-1, 1])).sum().item()
      tot += logits.size(0)
  return hit / tot

def get_loss(model, dl):
  criterion = torch.nn.CrossEntropyLoss(reduction='none')
  loss = []

  model.eval()
  with torch.no_grad():
    for i, (data, label) in enumerate(dl):
      data = data.cuda(non_blocking=True); label = label.cuda(non_blocking=True).long()
      logits = model(data)
      loss.append(criterion(logits, label))
  return torch.cat(loss)

class FeatDataset(torch.utils.data.Dataset):
  def __init__(self, feats, labels):
    self.feats = torch.from_numpy(feats)
    self.labels = torch.from_numpy(labels)

  def __len__(self):
    return self.feats.shape[0]

  def __getitem__(self, idx):
    return self.feats[idx], self.labels[idx]

def ddp_setup():
  init_process_group(backend="nccl")

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def gen_pseudo_labels(model, pseudo_feats):
    pseudo_labels = []

    model.eval()
    with torch.no_grad():
      for data in batch(pseudo_feats, 256):
        data = torch.from_numpy(data).cuda(non_blocking=True)
        pseudo_labels.append(model(data).topk(k=1)[1].view(-1).cpu().numpy())
    pseudo_labels = np.concatenate(pseudo_labels)
    return pseudo_labels

def gen_label_prob(model, pseudo_feats):
    labels_prob = []
    pseudo_labels = []

    model.eval()
    with torch.no_grad():
      for data in batch(pseudo_feats, 256):
        data = torch.from_numpy(data).cuda(non_blocking=True)
        labels_prob.append(torch.nn.functional.softmax(model(data), dim=1))
        pseudo_labels.append(model(data).topk(k=1)[1].view(-1).cpu().numpy())

    labels_prob = torch.vstack(labels_prob)
    pseudo_labels = np.concatenate(pseudo_labels)
    return labels_prob, pseudo_labels

def diverse_sample(X, K, Candidate, Labeled):
    embs = torch.Tensor(X).cuda()
    
    labeled_mu = []
    for item in Labeled:
      if len(labeled_mu) == 0:
        D2 = torch.cdist(embs[item].view(1,-1), embs, 2)[0]
      else:
        newD = torch.cdist(labeled_mu[-1].view(1,-1), embs, 2)[0]
        D2 = torch.min(D2, newD)

    ind = Candidate[0]
    mu = [embs[ind]]
    indsAll = [ind]
    while len(mu) < K:
        newD = torch.cdist(mu[-1].view(1,-1), embs, 2)[0]
        D2 = torch.min(D2, newD)
        current_D2 = D2.clone().cpu().numpy()
        if sum(current_D2) == 0.0: pdb.set_trace()
        current_D2 = current_D2.ravel().astype(float)
        Ddist = (current_D2 ** 2)/ sum(current_D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(current_D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ((ind in indsAll) or (ind not in Candidate)): ind = customDist.rvs(size=1)[0]
        mu.append(embs[ind])
        indsAll.append(ind)
    return indsAll

