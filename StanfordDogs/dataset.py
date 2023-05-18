import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

class StanfordDataset(Dataset): 
  def __init__(self, src):

    self.__num_classes = 120

    self.normalize = transforms.Normalize(
      mean=[0.5, 0.5, 0.5],
      std=[0.5, 0.5, 0.5]
      )

    self.transform = transforms.Compose(
      [transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       self.normalize,
       ])

    self.dataset = ImageFolder(
      src,
      transform=self.transform,
      target_transform=transforms.Compose([
        lambda x:torch.tensor([x]),
        lambda x:torch.FloatTensor(np.array(F.one_hot(x, self.__num_classes)))
        ])
      )

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.dataset)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    return self.dataset[idx]