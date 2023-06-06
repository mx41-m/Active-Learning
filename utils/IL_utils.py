import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

def cal_single_grad(model, single_data, single_label):
  criterion = torch.nn.CrossEntropyLoss()

  model.train()
  for name, module in model.named_modules():
    for param in module.parameters():
      param.requires_grad = True

  model.zero_grad()
  logits = model(single_data)
  loss = criterion(logits, single_label)
  loss.mean().backward()
  grads = []
  for param in model.parameters():
    grads.append(param.grad.clone())
  model.zero_grad()
  return grads

def cal_grad(model, test_dataloader):
  criterion = torch.nn.CrossEntropyLoss(reduction='none')

  model.train()
  model.zero_grad()
  for name, module in model.named_modules():
    for param in module.parameters():
      param.requires_grad = True

  all_grads = []
  for param in model.parameters():
    all_grads.append(torch.zeros(param.size()).cuda())

  for i, (data, label) in enumerate(test_dataloader):
    data = data.cuda(non_blocking=True); label = label.cuda(non_blocking=True).long()
    model.zero_grad()
    logits = model(data)
    loss = criterion(logits, label)
    loss.mean().backward()
    grads = []
    for i, param in enumerate(model.parameters()):
        all_grads[i] += param.grad.clone()
  model.zero_grad()
  return all_grads

def hessian_vector_product(model, criterion, s_test, data, label, weight_decay):
  ### first grad
  loss = criterion(model(data), label)
  param_list = [param for param in model.parameters()]
  reg = torch.cat([i.view(-1) ** 2 for i in param_list]).sum() / 2 * 1e-4
  grad = torch.autograd.grad(outputs = loss + reg, inputs = param_list, create_graph = True)
  grad = torch.cat([item.view(-1) for item in grad]).cuda()

  v_elem = torch.cat([item.view(-1) for item in s_test]).cuda()
  ### sceond grad
  elemwise_product = [x * y for x, y in zip(grad, v_elem)]
  elemgrad = torch.autograd.grad(outputs = elemwise_product, inputs = param_list, retain_graph = True)

  return elemgrad

def approximate_hessian_inverse(model, weight_decay, train_dataset, test_dataloader, r, recursion_depth, damping, scale, final_s_test):

  v = cal_grad(model, test_dataloader)
  criterion = torch.nn.CrossEntropyLoss()

  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=4)
  for i in tqdm(range(r)):
    s_test = [item.clone() for item in v]
    for ep in range(recursion_depth):
      for data, label in train_dataloader:
        data = data.cuda(non_blocking=True); label = label.cuda(non_blocking=True).long()
        hessian_vector_val = hessian_vector_product(model, criterion, s_test, data, label, weight_decay)
        s_test = [a + (1 - damping) * b - c / scale for (a, b, c) in zip(v, s_test, hessian_vector_val)]
    final_s_test[r*dist.get_rank()+i] = (torch.cat([item.view(-1) for item in s_test]))

  return final_s_test

