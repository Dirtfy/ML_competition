import os
import timm
import torch
import torch.nn as nn

import numpy as np

def save_train_result(model, path, name, optimizer, criterion, batch_size, shuffle, epoch):
    os.makedirs(path, exist_ok=False)

    f = open(path+"/"+name+".txt", "w")
    f.write("model: \n"+str(model)+"\n\n")
    f.write("optimizer: \n"+str(optimizer)+"\n\n")
    f.write("criterion: \n"+str(criterion)+"\n\n")
    f.write("batch_size: \n"+str(batch_size)+"\n\n")
    f.write("shuffle: \n"+str(shuffle)+"\n\n")
    f.write("epoch: \n"+str(epoch)+"\n\n")
    f.close()

    torch.save(model.state_dict(), path+"/"+name+".pt")

class StanfordModel(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        self.__num_classes = 120
        self.device = device

        self.backbone = timm.models.vit_base_patch16_224(pretrained=True).to(device)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.head.parameters():
            param.requires_grad = True

        self.head = nn.Sequential(
            nn.BatchNorm1d(num_features=1000),

            nn.Linear(
            in_features = 1000,
            out_features = 512
            ),

            # nn.BatchNorm1d(num_features=512),

            nn.ELU(),

            nn.Linear(
            in_features = 512,
            out_features = 256
            ),

            nn.ELU(),

            nn.Linear(
            in_features = 256,
            out_features = self.__num_classes
            ),

            nn.Softmax()
        ).to(device)

        self.params = list(self.backbone.head.parameters()) + list(self.head.parameters())

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
    
    def c_train(self, epoch, dataset, learning_rate, batch_size, shuffle, path, name, optimizer=None, criterion=None):
        if optimizer == None:
            optimizer = torch.optim.Adam(self.params, lr=learning_rate)
        if criterion == None:
            criterion = torch.nn.CrossEntropyLoss()

        dataloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=1)
        
        for ep in range(epoch):   # 데이터셋을 수차례 반복합니다.

            epoch_cnt = 0
            epoch_cor_cnt = 0
            epoch_loss_sum = 0.0
            for j, datas in enumerate(dataloader):
                self.train()
                
                # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
                images, labels = datas
                images = images.to(self.device)  # [100, 3, 224, 224]
                labels = labels.to(self.device)  # [100]

                # 변화도(Gradient) 매개변수를 0으로 만들고
                optimizer.zero_grad()

                # 순전파 + 역전파 + 최적화를 한 후
                outputs = self.forward(images)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    self.eval()

                    batch_cnt = 0
                    batch_cor_cnt = 0
                    
                    for i, image in enumerate(datas[0]):
                        batch_cnt+=1

                        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
                        image = image.to(self.device)  # [100, 3, 224, 224]
                        label = datas[1][i]
                        label = label.to(self.device)  # [100]

                        # 순전파 + 역전파 + 최적화를 한 후
                        output = self.forward(image[None, ...])
                        output = output.squeeze()
                        label = label.squeeze()

                        pred = torch.argmax(output)
                        batch_cor_cnt += label[pred]

                epoch_cnt += batch_cnt
                epoch_cor_cnt += batch_cor_cnt                   

                # 통계를 출력합니다.
                epoch_loss_sum += loss.item()
                
                print(f'[epoch: {ep + 1}, batch: {j + 1:5d}] loss: {loss.item()}, batch_acc: {batch_cor_cnt/batch_cnt}')

            print(f'[epoch: {ep + 1}] epoch_loss_sum: {epoch_loss_sum}, epoch_acc: {epoch_cor_cnt/epoch_cnt}')

        save_train_result(self, path, name, optimizer, criterion, batch_size, shuffle, epoch)

    def test(self, dataset, criterion=None):
        if criterion == None:
            criterion = torch.nn.CrossEntropyLoss()

        correct_top1 = 0
        loss_sum = 0
        total_cnt = 0

        self.eval()
        with torch.no_grad():
            for i, data in enumerate(dataset):
                # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
                image, label = data
                image = image.to(self.device)  # [100, 3, 224, 224]
                label = label.to(self.device)  # [100]

                # 순전파 + 역전파 + 최적화를 한 후
                output = self.forward(image[None, ...])
                output = output.squeeze()
                label = label.squeeze()
                loss = criterion(output, label)
                loss_sum += loss

                pred = torch.argmax(output)
                correct_top1 += label[pred]
                
                total_cnt += 1
                print(f'[{i + 1}] accuracy: {correct_top1/total_cnt}, loss: {loss.item():}')

        print(f'accuracy : {correct_top1/total_cnt}, average loss : {loss_sum/total_cnt}')

    def load_weight(self, path):
        self.load_state_dict(torch.load(path))

    