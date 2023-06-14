from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset

import util

class StanfordDataset(Dataset): 
  def __init__(self, src, resol, augmente=False, augmente_time=10):

    self.normalize = transforms.Normalize(
      mean=[0.5, 0.5, 0.5],
      std=[0.5, 0.5, 0.5]
      )
    
    self.input_transform = transforms.Compose([
      transforms.Resize(int(resol/0.875)),
      transforms.CenterCrop(resol),
      transforms.ToTensor(),
      self.normalize
      ])

    self.target_transfrom = transforms.Compose([
        transforms.Lambda(util.toTensor),
        transforms.Lambda(util.toOne_hot),
        ])
    
    if augmente:
      self.origin_dataset = ImageFolder(
        src,
        transform=self.input_transform,
        target_transform=self.target_transfrom
        )
      
      self.input_augmente = transforms.Compose([
            transforms.Resize(int(resol/0.875)),
            transforms.RandomCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, shear=0.05, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.RandomResizedCrop(resol, scale=(0.8, 1.2), ratio=(0.75, 1.33)),
            transforms.ToTensor(),
            self.normalize
        ])
      
      self.augmented_datasets = []
      for _ in range(augmente_time):
        self.augmented_datasets += [
          ImageFolder(
          src,
          transform=self.input_augmente,
          target_transform=self.target_transfrom
          )
        ]
      
      self.dataset = ConcatDataset([self.origin_dataset] + self.augmented_datasets)
    else:
      self.dataset = ImageFolder(
        src,
        transform=self.input_transform,
        target_transform=self.target_transfrom
        )

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.dataset)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    return self.dataset[idx]