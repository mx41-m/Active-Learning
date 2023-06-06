from torchvision import transforms

datasets_transforms = {
'pre-ResNet18':{
  'cifar10':{
    'train': transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomResizedCrop(224),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ]), 
    'test': transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ]),
    },
  'cifar100':{
    'train': transforms.Compose([    
      transforms.RandomHorizontalFlip(),
      transforms.RandomResizedCrop(224),
      transforms.ToTensor(), 
      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
      ]),
    'test': transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),  
      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
      ]),
    },
  'iNaturalist':{
    'train': transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomResizedCrop(224),
      transforms.ToTensor(),  
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ]),
    'test': transforms.Compose([
      transforms.Resize(256), 
      transforms.CenterCrop(224),
      transforms.ToTensor(),  
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ])
    }
   },
'scratch-ResNet18':{
  'cifar10':{
     'train': transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4),
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
       ]),
     'test': transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
       ])
     }
  },
'clip': {
    'iNaturalist':{
      'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
      'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
       ])
      }
  }
}

