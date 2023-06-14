import timm
import torch
import torch.nn as nn
import copy

from . import util
from . import stopper

class StanfordModel(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        # 클래스 개수
        self.__num_classes = 120
        self.device = device

        self.backbone = timm.create_model('convit_base', pretrained=True).to(device)
        # self.backbone = timm.create_model('deit_base_distilled_patch16_384', pretrained=True).to(device)
        # self.backbone = timm.create_model('resnet200d', pretrained=True).to(device)
        self.head = nn.Sequential(
            nn.BatchNorm1d(num_features=1000),
            nn.Linear(in_features=1000, out_features=512),
            nn.ELU(),
            nn.Dropout(),

            nn.BatchNorm1d(num_features=512),
            nn.Linear(in_features=512, out_features=256),
            nn.ELU(),
            nn.Dropout(),

            nn.Linear(in_features=256, out_features=self.__num_classes)
        ).to(device)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
    
    def train_loop(self, epoch, train_set, test_set, path, name, learning_rate=0.00005, batch_size=16, shuffle=True, optimizer=None, criterion=None, scheduler=None, ls=0.3, wd=0.0, pt=3):
        if optimizer == None:
            optimizer = torch.optim.RAdam(self.parameters(), lr=learning_rate, weight_decay=wd)
        if criterion == None:
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=ls)
        if scheduler == None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=pt, cooldown=pt)

        dataloader = torch.utils.data.DataLoader(train_set,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=4)

        early = stopper.EarlyStop(pt)

        epoch_cnt = len(train_set)
        
        best_acc = 0
        best_ep = 0
        for ep in range(epoch):            
            epoch_cor_cnt = 0
            epoch_loss_sum = 0.0

            datas_list = []
            avg_loss_list = []
            for j, datas in enumerate(dataloader):
                datas_list += [datas]

                self.train_step(datas, optimizer, criterion)

                # 배치 결과 계산
                tmp_dataset = [(datas[0][i], datas[1][i]) for i in range(len(datas[0]))]
                batch_loss, batch_cor, batch_cnt = self.test(tmp_dataset)

                avg_loss_list += [batch_loss/batch_cnt]

                # 배치 결과 합산
                epoch_cor_cnt += batch_cor
                epoch_loss_sum += batch_loss
                # 배치 결과 출력
                print(f'[epoch: {ep + 1}, batch: {j + 1:5d}], batch_avg_loss: {batch_loss/batch_cnt}, batch_acc: {batch_cor/batch_cnt}, train_acc: {epoch_cor_cnt/epoch_cnt}')
            
            # learning rate 출력
            print('lr: ', optimizer.param_groups[0]['lr'])

            # 에포크 결과 출력
            print(f'[epoch: {ep + 1}] train_avg_loss: {epoch_loss_sum/epoch_cnt}, train_acc: {epoch_cor_cnt/epoch_cnt}')
            early.train_loss_list += [epoch_loss_sum/epoch_cnt]
            early.train_acc_list += [epoch_cor_cnt/epoch_cnt]

            bad_datas = []
            for j, data in enumerate(datas_list):
                if avg_loss_list[j] >= (epoch_loss_sum/epoch_cnt)*2:
                    bad_datas += [data]

            bad_repeat = 100
            for j, datas in enumerate(bad_datas):
                for k in range(bad_repeat):
                    self.train_step(datas, optimizer, criterion)

                    # 배치 결과 계산
                    tmp_dataset = [(datas[0][i], datas[1][i]) for i in range(len(datas[0]))]
                    batch_loss, batch_cor, batch_cnt = self.test(tmp_dataset)

                    # 배치 결과 출력
                    print(f'[epoch: {ep + 1}, batch: {j + 1:5d}, bad_repeat: {k + 1:5d}], batch_avg_loss: {batch_loss/batch_cnt}, batch_acc: {batch_cor/batch_cnt}')

            # test_dataset 결과 계산 및 출력
            print('evaluating...')
            test_loss, test_cor, test_cnt = self.test(test_set)
            print(f'[epoch: {ep + 1}] test_avg_loss: {test_loss/test_cnt}, test_acc: {test_cor/test_cnt}')
            early.test_loss_list += [test_loss/test_cnt]
            early.test_acc_list += [test_cor/test_cnt]

            scheduler.step()

            # best model 기록
            if test_cor/test_cnt > best_acc:
                best_ep = ep
                best_acc = test_cor/test_cnt
                best_acc_model = copy.deepcopy(self.state_dict())

            # early stop
            if early.stop():
                epoch = ep
                break

        # 모델 저장
        util.save_train_result(self, path, name+'_end', optimizer, criterion, batch_size, shuffle, epoch, best_ep, early, ls, wd, pt)   
        torch.save(best_acc_model, path+"/"+name+"_best.pt") 

    def train_step(self, datas, optimizer, criterion):
        self.train()

        images, labels = datas
        images = images.to(self.device)
        labels = labels.to(self.device)

        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화
        outputs = self.forward(images)
        loss = criterion(outputs.squeeze(), labels.squeeze())
        loss.backward()
        optimizer.step()

    def test(self, dataset, criterion=None, prt=False):
        if criterion == None:
            criterion = torch.nn.CrossEntropyLoss()

        correct_top1 = 0
        loss_sum = 0
        total_cnt = 0

        self.eval()
        with torch.no_grad():
            for i, data in enumerate(dataset):

                image, label = data
                image = image.to(self.device)
                label = label.to(self.device)

                # 순전파 + loss 계산
                output = self.forward(image[None, ...])
                output = output.squeeze()
                label = label.squeeze()
                loss = criterion(output, label)
                loss_sum += loss.item()

                # top-1 확인
                pred = torch.argmax(output)
                correct_top1 += label[pred]
                
                total_cnt += 1

                if prt:
                    print(f'[{i + 1}] loss: {loss.item():}, accuracy: {correct_top1/total_cnt}')

        if prt:
            print(f'average loss : {loss_sum/total_cnt}, accuracy : {correct_top1/total_cnt}')

        return loss_sum, correct_top1, total_cnt

    def load_weight(self, path):
        self.load_state_dict(torch.load(path))

    