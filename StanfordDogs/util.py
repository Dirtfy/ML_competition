import os
import shutil
import torch
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F

def path(*names):
    path = ""
    for name in names:
        path = path+name+'/'
    return path[:-1]

def copyTo(image, label, dst):
  os.makedirs(path(dst, label), exist_ok=True)
  shutil.copy2(image, path(dst, label))

def save_train_result(model, path, name, optimizer, criterion, batch_size, shuffle, epoch, early, ls, wd, pt):
    os.makedirs(path, exist_ok=False)

    f = open(path+"/"+name+".txt", "w")
    f.write("model: \n"+str(model)+"\n\n")
    f.write("optimizer: \n"+str(optimizer)+"\n\n")
    f.write("weight decay: \n"+str(wd)+"\n\n")
    f.write("criterion: \n"+str(criterion)+"\n\n")
    f.write("label smoothing: \n"+str(ls)+"\n\n")
    f.write("patience: \n"+str(pt)+"\n\n")
    f.write("batch_size: \n"+str(batch_size)+"\n\n")
    f.write("shuffle: \n"+str(shuffle)+"\n\n")
    f.write("epoch: \n"+str(epoch)+"\n\n")
    f.close()

    torch.save(model.state_dict(), path+"/"+name+".pt")

    test_acc_list = detach(early.test_acc_list)
    train_acc_list = detach(early.train_acc_list)
    test_loss_list = detach(early.test_loss_list)
    train_loss_list = detach(early.train_loss_list)

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(test_acc_list)+1, 1), test_acc_list, label='test')
    plt.plot(range(1, len(train_loss_list)+1, 1), train_acc_list, label='train')
    plt.legend()
    plt.title('Acc')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(test_loss_list)+1, 1), test_loss_list, label='test')
    plt.plot(range(1, len(train_loss_list)+1, 1), train_loss_list, label='train')
    plt.legend()
    plt.title('Loss')

    plt.savefig(path+"/"+name+".png")

# 학습률 조정 함수
def lr_func(epoch):
    if epoch < 3:
        return 1
    else:
        return (0.75 ** (epoch-2))
    
def detach(list):
    return [v.cpu().numpy() if torch.is_tensor(v) else v for v in list]

def toTensor(x):
  return torch.tensor([x])
def toOne_hot(x):
  return torch.FloatTensor(np.array(F.one_hot(x, 120)))

def weight_s_sum(model):
    sum = 0
    for param in model.parameters():
        sum += torch.sum(param.data()) ** 2
    return sum
