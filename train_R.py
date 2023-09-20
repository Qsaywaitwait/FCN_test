import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data

from FCNtest.FCN8S_R import FCN8s
from FCNtest.dataset_R import MyDataset
from FCNtest.evaluation_R import Iou, miou
from FCNtest.utils_R import draw_pic

'''model:模型, criterion损失函数， optimizer:优化方法, traindataloader:训练集, valdataloader：验证集'''
def train_model(model, criterion, optimizer, traindataloader, valdataloader, num_epochs):
    for epoch in range(num_epochs):
        train_loss_rec = []
        acc_rec = []
        print('Eopch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_loss = 0.
        train_num = 0
        # 训练
        model.train()
        for step, (b_x, b_y) in enumerate(traindataloader):
            optimizer.zero_grad()
            b_x = b_x.float().to(device)  # [BATCHSIZE, 3, 320, 480]
            b_y = b_y.long().to(device)  # [BATCHSIZE, 320, 480]
            out = model(b_x)
            # print('out1.shape{}'.format(out.shape))  #torch.Size([2, 21, 320, 480])
            out = F.log_softmax(out, dim=1)
            pre_lab = torch.argmax(out, 1)  # pre_lab.shape = [BATCHSIZE, 320, 480]

            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            # print('out.shape{}'.format(out.shape))  # out.shape torch.Size([2, 21, 320, 480])
            # print('by.shape{}'.format(b_y.shape))  # torch.Size([2, 320, 480])

            train_loss += loss.item() * len(b_y)
            train_num += len(b_y)
            a = (train_loss / train_num)
            a=format(a, '.5f')
            train_loss_rec.append(a)
            # 计算PA
            train_correct = torch.sum(pre_lab == b_y.data) / (BATCHSIZE * high * width)
            # train_correct= format(train_correct, '.5f')
            acc_rec.append(train_correct)
            # torch.save(model.state_dict(), 'fcn8s.pkl')
            # 可视化训练效果
            if step% 1000 ==0:
                draw_pic(b_x, b_y, pre_lab)
                torch.save(fcn8s.state_dict(), 'fcn8s.pkl')
                # print('draw_finished')
        # acc_rec = sum(acc_rec)/len(acc_rec)
        # acc_rec = format(acc_rec, '.5f')
        epoch_loss = train_loss/train_num
        epoch_loss = format(epoch_loss, '.5f')
        print('epoch:{} | mean loss:{}'.format(epoch, epoch_loss))
        torch.save(model.state_dict(), 'fcn8s.pkl')


        # 验证：
        val_loss = 0.
        val_num = 0
        model.eval()
        MIOU = []
        MIOU_RR = []
        LOSS = []
        VC = []
        for step, (b_x, b_y) in enumerate(valdataloader):
            b_x = b_x.float().to(device)   # [BATCHSIZE, 3, 320, 480]
            b_y = b_y.long().to(device)  # [BATCHSIZE, 320, 480]
            out = model(b_x)
            out = F.log_softmax(out, dim=1)
            pre_lab = torch.argmax(out, 1)
            loss = criterion(out, b_y)
            val_loss += loss.item() * len(b_y)
            val_num += len(b_y)
            val_correct = torch.sum(pre_lab == b_y.data) / (BATCHSIZE * high * width)
            MIOU.append(Iou(pre_lab, b_y, 21))
            MIOU_RR.append(miou(pre_lab, b_y, 21))
            LOSS.append(loss.item())
            VC.append(val_correct)
            # 可视化训练效果
            # print('epoch:{} | step:{} | val loss:{:.5f} | PA:{:.5f} | MIOU:{:.5f}'.format(epoch, step, loss.item(),
            #                                                                               val_correct,
            #                                                                               Iou(pre_lab, b_y, 21)))
        LOSSa = sum(LOSS) / len(LOSS)
        VCa = sum(VC)/len(VC)
        MIOUa = sum(MIOU) / len(MIOU)
        print('epoch:{} | val loss:{:.5f} | PA:{:.5f} | MIOU:{:.5f}'.format(epoch, LOSSa, VCa, MIOUa))

    return model



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 给出基本数据
high, width = 320, 480
EPOCH = 1
BATCHSIZE = 2
LR = 5e-4
# 导入数据集
# voc_train = MyDataset(
#     "E:/deeplearning/dataB/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt", high, width)
# voc_val = MyDataset("E:/deeplearning/dataB/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
#                     high, width)
# train_loader = data.DataLoader(voc_train, batch_size=BATCHSIZE, shuffle=True)
# val_loader = data.DataLoader(voc_val, batch_size=BATCHSIZE, shuffle=True)
# print('训练集大小:{}'.format(voc_train.__len__()))
# print('验证集大小:{}'.format(voc_val.__len__()))

# 导入网络:
voc_train = MyDataset(
    "E:/deeplearning/dataB/VOCtrainval_11-May-2012_B/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt", high, width)
voc_val = MyDataset("E:/deeplearning/dataB/VOCtrainval_11-May-2012_B/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
                    high, width)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = data.DataLoader(voc_train, batch_size=BATCHSIZE, shuffle=True)
val_loader = data.DataLoader(voc_val, batch_size=BATCHSIZE, shuffle=True)
print('训练集大小:{}'.format(voc_train.__len__()))
print('验证集大小:{}'.format(voc_val.__len__()))
fcn8s = FCN8s(21).to(device)
fcn8s.load_state_dict(torch.load('fcn8s.pkl'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fcn8s.parameters(), lr=LR, weight_decay=1e-4)
# 迭代训练：
'''model:模型, criterion损失函数， optimizer:优化方法, traindataloader:训练集, valdataloader：验证集'''
fcn8s = train_model(fcn8s, criterion, optimizer, train_loader, val_loader, EPOCH)
torch.save(fcn8s.state_dict(), 'fcn8s.pkl')