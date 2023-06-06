# Active-Learning
Implementation of the RALIF active learning algorithm. 

# Dependencies
To run the codes, you need PyTorch (2.0.0)

# Running an experiment
To run active learning selection on default AL selection scenario, where dataset is Cifar10, model is pretrain ResNet18 and badge size is 1000
python -m torch.distributed.launch --nproc_per_node num_gpus --use_env al_train.py
