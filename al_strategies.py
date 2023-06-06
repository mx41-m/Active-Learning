import torch
import numpy as np
from utils.utils import *
from utils.IL_utils import *
from extract_feat import extract_features_func
from tqdm import tqdm
import torch.nn as nn
import os
import torch.distributed as dist

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import resnet
from torch.distributions import Categorical

def ralif_query(args, labeled_set, total_train_feats, total_train_labels, iteration, select_pool):

    WORLD_SIZE = dist.get_world_size()
    ### multi gpu
    all_influence = torch.zeros([len(select_pool)], device='cuda')
    if os.path.exists(args.res_ck + 'results/influence_%d.pt' % (iteration)):
      all_influence = torch.load(args.res_ck + 'results/influence_%d.pt' % (iteration))
      labels_prob = torch.load(args.res_ck + 'results/labels_prob_%d.pt' % (iteration))
      pseudo_labels = torch.load(args.res_ck + 'results/pseudo_labels_%d.pt' % (iteration))

    if len(torch.where(all_influence!=0)[0]) != len(select_pool):
      train_dataset = FeatDataset(total_train_feats[labeled_set], total_train_labels[labeled_set])
      train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle=False, sampler=DistributedSampler(train_dataset))
      if args.model == 'pre-ResNet18' or args.model == 'scratch-ResNet18':
        model = CustomLogisticRegression(args.feat_dim, args.cls_num).cuda()
      else:
        model = CustomLogisticRegression(args.feat_dim, args.cls_num).cuda()

      if os.path.exists(args.res_ck + 'models/models_%d.pt' % (iteration)):
        model.fc.weight.data[...] = torch.load(args.res_ck + 'models/models_%d.pt' % (iteration))['weight']
        model.fc.bias.data[...] = torch.load(args.res_ck + 'models/models_%d.pt' % (iteration))['bias']
        model = DDP(model, device_ids=[dist.get_rank()])
      else:
        model = DDP(model, device_ids=[dist.get_rank()])
        model = ddp_train_model(model, train_dataloader, args.epoch, args.lr, 1e-4)

      labels_prob, pseudo_labels = gen_label_prob(model, total_train_feats[select_pool])

      if dist.get_rank() == 0:
        torch.save(labels_prob, args.res_ck + 'results/labels_prob_%d.pt' % (iteration))
        torch.save(pseudo_labels, args.res_ck + 'results/pseudo_labels_%d.pt' % (iteration))
        torch.save(model.state_dict(), args.res_ck + 'results/model.pt')

      val_dataset = FeatDataset(total_train_feats[select_pool], pseudo_labels)
      val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 256, shuffle=False, num_workers=4)
      
      ### generate IL with HVP
      if os.path.exists(args.res_ck + 'results/hessian_%d.pt' % (iteration)):
        hv = torch.load(args.res_ck + 'results/hessian_%d.pt' % (iteration)).cuda()
      else:
        r, recursion_depth = 20, 50
        hessian_inverse = torch.zeros([r * WORLD_SIZE, args.feat_dim * args.cls_num + args.cls_num], device='cuda')
        dist.broadcast(hessian_inverse, src=0)
        hessian_inverse = approximate_hessian_inverse(model, 1e-4, train_dataset, val_dataloader, r, recursion_depth,  0.01, 25, hessian_inverse)
        dist.all_reduce(hessian_inverse)
        hv = hessian_inverse.mean(dim=0)
        if dist.get_rank() == 0:
          torch.save(hv, args.res_ck + 'results/hessian_%d.pt' % (iteration))

      unlabeled_dataset = FeatDataset(total_train_feats[select_pool], pseudo_labels)
      unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=1, shuffle=False, sampler=range(dist.get_rank(), len(unlabeled_dataset), dist.get_world_size()))
      val_loss = get_loss(model, val_dataloader)
      all_influence = torch.zeros([len(unlabeled_dataset), args.cls_num], device='cuda')
      dist.broadcast(all_influence, src=0)

      for idx, (data, pseudo_label) in tqdm(enumerate(unlabeled_dataloader)):
          data = data.cuda(non_blocking=True)
          pseudo_label = pseudo_label.cuda(non_blocking=True).long()

          for each_label in range(args.cls_num):
            with model.no_sync():
              grad = cal_single_grad(model, data, torch.tensor(each_label).reshape(1).cuda(non_blocking=True).long())
            grad = torch.cat([item.view(-1) for item in grad])
            single_label_influence = torch.sum(val_loss) - torch.matmul(grad, hv) / len(labeled_set)
            all_influence[dist.get_rank():len(unlabeled_dataset):dist.get_world_size()][idx, each_label] = single_label_influence

      dist.all_reduce(all_influence)
      if dist.get_rank() == 0:
        torch.save(all_influence, args.res_ck + 'results/influence_%d.pt' % (iteration))

    ### truncated IL
    if dist.get_rank() == 0:
      print('Truncated-IL')
    exp_influence = []
    for idx, prob in enumerate(labels_prob):
      most_label = np.where(prob.cpu() >  (args.tau/args.cls_num))[0]
      exp_influence.append(torch.mean(labels_prob[idx][most_label] * all_influence[idx][most_label]))
    exp_influence = torch.tensor(exp_influence)

    ### diverse sampling
    if dist.get_rank() == 0:
      print('Diverse-sampling')
    sel_candidate = torch.argsort(exp_influence)[:int(len(exp_influence) * 0.1)].numpy()
    diverse = torch.tensor(diverse_sample(total_train_feats, args.budget_size, select_pool[sel_candidate], labeled_set)).cuda()

    dist.barrier()
    dist.broadcast(diverse, src=0)
    return diverse.cpu().numpy()

