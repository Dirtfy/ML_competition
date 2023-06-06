import timm
import torch
import torch.nn as nn

import util
import stopper

class StanfordModel(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        # 클래스 개수
        self.__num_classes = 120
        self.device = device

        # model: vit_base_patch16_224
        self.backbone = timm.create_model('resnet200d', pretrained=True).to(device)
        # self.backbone.fc = nn.Linear(in_features=2048, out_features=self.__num_classes, bias=True).to(device)
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
        # self.backbone.head = nn.Linear(in_features=768, out_features=self.__num_classes).to(device)

        # 전체 학습
        # for param in self.backbone.parameters():
        #     param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
    
    def c_train(self, epoch, train_set, test_set, learning_rate, batch_size, shuffle, path, name, optimizer=None, criterion=None):
        if optimizer == None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if criterion == None:
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.3)

        dataloader = torch.utils.data.DataLoader(train_set,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=4)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = util.lr_func)

        early = stopper.EarlyStop(10)

        epoch_cnt = len(train_set)
        
        for ep in range(epoch):   # 데이터셋을 수차례 반복합니다.
            self.train()
            
            epoch_cor_cnt = 0
            epoch_loss_sum = 0.0
            for j, datas in enumerate(dataloader):
                self.train()
                
                # [inputs, labels]의 목록인 data로부터 입력
                images, labels = datas
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 변화도(Gradient) 매개변수를 0으로 만들고
                optimizer.zero_grad()

                # 순전파 + 역전파 + 최적화
                outputs = self.forward(images)
                # print(outputs.squeeze())
                # print(labels.squeeze())
                loss = criterion(outputs.squeeze(), labels.squeeze())
                loss.backward()
                optimizer.step()

                # 배치 결과 계산 및 출력
                tmp_dataset = [(datas[0][i], datas[1][i]) for i in range(len(datas[0]))]
                batch_avg_loss, batch_acc, batch_cnt = self.test(tmp_dataset)
                print(f'[epoch: {ep + 1}, batch: {j + 1:5d}], batch_avg_loss: {batch_avg_loss}, batch_acc: {batch_acc}')

                # 배치 결과 합산
                epoch_cor_cnt += batch_cnt
                epoch_loss_sum += loss.item()

            scheduler.step()
            # learning rate 출력
            print('lr: ', optimizer.param_groups[0]['lr'])

            # 에포크 결과 출력
            print(f'[epoch: {ep + 1}] train_avg_loss: {epoch_loss_sum/epoch_cnt}, train_acc: {epoch_cor_cnt/epoch_cnt}')
            early.train_loss_list += [epoch_loss_sum/epoch_cnt]
            early.train_acc_list += [epoch_cor_cnt/epoch_cnt]

            # test_dataset 결과 계산 및 출력
            print('evaluating...')
            test_avg_loss, test_acc, _ = self.test(test_set)
            print(f'[epoch: {ep + 1}] test_avg_loss: {test_avg_loss}, test_acc: {test_acc}')
            early.test_loss_list += [test_avg_loss]
            early.test_acc_list += [test_acc]

            if early.stop():
                break

        # 모델 저장
        util.save_train_result(self, path, name, optimizer, criterion, batch_size, shuffle, epoch, early)        

    def test(self, dataset, criterion=None, prt=False):
        if criterion == None:
            criterion = torch.nn.CrossEntropyLoss()

        correct_top1 = 0
        loss_sum = 0
        total_cnt = 0

        self.eval()
        with torch.no_grad():
            for i, data in enumerate(dataset):
                # [inputs, labels]의 목록인 data로부터 입력
                image, label = data
                image = image.to(self.device)
                label = label.to(self.device)

                # 순전파 + loss 계산
                output = self.forward(image[None, ...])
                output = output.squeeze()
                label = label.squeeze()
                loss = criterion(output, label)
                loss_sum += loss

                # top-1 확인
                pred = torch.argmax(output)
                correct_top1 += label[pred]
                
                total_cnt += 1

                if prt:
                    print(f'[{i + 1}] loss: {loss.item():}, accuracy: {correct_top1/total_cnt}')

        if prt:
            print(f'average loss : {loss_sum/total_cnt}, accuracy : {correct_top1/total_cnt}')

        return loss_sum/total_cnt, correct_top1/total_cnt, correct_top1

    def load_weight(self, path):
        self.load_state_dict(torch.load(path))

    