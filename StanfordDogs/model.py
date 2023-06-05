import os
import timm
import torch
import torch.nn as nn

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

# 학습률 조정
def lr_func(epoch):
    if epoch < 3:
        return 1
    else:
        return (0.95 ** (epoch-2))

class StanfordModel(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        # 클래스 개수
        self.__num_classes = 120
        self.device = device

        # model: vit_base_patch16_224
        self.model = timm.models.vit_base_patch16_224(pretrained=False).to(device)
        self.model.head = nn.Linear(in_features=768, out_features=self.__num_classes).to(device)

        # 전체 학습
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
    
    def c_train(self, epoch, train_set, test_set, learning_rate, batch_size, shuffle, path, name, optimizer=None, criterion=None):
        if optimizer == None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if criterion == None:
            criterion = torch.nn.CrossEntropyLoss()

        dataloader = torch.utils.data.DataLoader(train_set,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=1)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_func)

        epoch_cnt = len(train_set)
        
        pre_test_acc = 0
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

            # test_dataset 결과 계산 및 출력
            print('evaluating...')
            test_avg_loss, test_acc, _ = self.test(test_set)
            print(f'[epoch: {ep + 1}] test_avg_loss: {test_avg_loss}, test_acc: {test_acc}')

            if pre_test_acc/5 > test_acc:
                epoch = ep
                break
            pre_test_acc += test_acc
            if ep%5 == 0:
                pre_test_acc = 0

        # 모델 저장
        save_train_result(self, path, name, optimizer, criterion, batch_size, shuffle, epoch)

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

    