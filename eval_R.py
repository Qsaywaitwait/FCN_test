import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from FCNtest.FCN8S_R import FCN8s
from FCNtest.dataset_R import MyDataset
from FCNtest.evaluation_R import Iou, epoch_miou, mean_iou

'''model:模型, criterion损失函数， optimizer:优化方法, traindataloader:训练集, valdataloader：验证集'''
def train_model(model, criterion, optimizer, traindataloader, valdataloader, num_epochs):

    iou_rec =[]
    for epoch in range(num_epochs):
        # print('Eopch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        model.eval()

        eiou =[]
        for step , (b_x, b_y) in enumerate(valdataloader):
            b_x = b_x.float().to(device)   # [BATCHSIZE, 3, 320, 480]
            b_y = b_y.long().to(device)  # [BATCHSIZE, 320, 480]
            out = model(b_x)
            out = F.log_softmax(out, dim=1)
            pre_lab = torch.argmax(out, 1)
            # iou = Iou(pre_lab, b_y, 21)
            iou = Iou(b_y, pre_lab, 21)
            eiou = epoch_miou(eiou,iou)
        iou_rec = epoch_miou(iou_rec,eiou)
    ans = mean_iou(iou_rec)
    print('MIOU:{:.5f}'.format(ans))
    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 给出基本数据
high, width = 320, 480
EPOCH = 1
BATCHSIZE = 2
LR = 5e-4
# 导入数据集
voc_train = MyDataset(
    "E:/deeplearning/dataB/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt", high, width)
voc_val = MyDataset("E:/deeplearning/dataB/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
                    high, width)
train_loader = data.DataLoader(voc_train, batch_size=BATCHSIZE, shuffle=True)
val_loader = data.DataLoader(voc_val, batch_size=BATCHSIZE, shuffle=True)
print('训练集大小:{}'.format(voc_train.__len__()))
print('验证集大小:{}'.format(voc_val.__len__()))

# 导入网络:
fcn8s = FCN8s(21).to(device)
fcn8s.load_state_dict(torch.load('fcn8s.pkl'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fcn8s.parameters(), lr=LR, weight_decay=1e-4)
# 迭代训练：
'''model:模型, criterion损失函数， optimizer:优化方法, traindataloader:训练集, valdataloader：验证集'''
fcn8s = train_model(fcn8s, criterion, optimizer, train_loader, val_loader, EPOCH)