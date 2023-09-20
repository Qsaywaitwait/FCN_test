import numpy as np
import torch
from colorcet import cm
from visdom import Visdom
import random

# 将标签转化为图像
def label2image(prelabel):
    h, w = prelabel.shape
    prelabel = prelabel.reshape(h * w, -1)
    image = np.zeros((h * w, 3), dtype='int32')
    for ii in range(21):  # 共21个类别
        index = np.where(prelabel == ii)  # 找到n维数组中特定数值的下标
        image[index, :] = cmode(ii)

    return image.reshape(h, w, 3)

#画框取色函数
def cmode(param):
    if param==0:
        return(0,0,0)
    cmap = []
    random.seed(int(param))
    rand = random.random()
    color = list(cm.rainbow(rand))

    for i in range(3):
        cmap.append(int(color[i]*255))
    return tuple(cmap)
#去标准化
def inv_normalize_image(data):
    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std = np.array([0.229, 0.224, 0.225])
    data = data.astype('float32') * rgb_std + rgb_mean
    return data.clip(0 ,1)

# torch.Size([2, 320, 480]) | b_y:torch.Size([2, 320, 480]) |b_x:torch.Size([2, 3, 320, 480])
#          pre_lab:(320, 480, 3) | b_y:(320, 480, 3) |b_x:(320, 480, 3)

b_x = torch.rand(2, 3, 320, 480)
b_y = torch.rand(2, 320, 480)
pre_lab = torch.rand(2, 320, 480)
batchsize = 2


def draw_pic(b_x,b_y,pre_lab):
    b_x, b_y, pre_lab=b_x.to('cpu'),b_y.to('cpu'),pre_lab.to('cpu')
    b_x_numpy = b_x.data.numpy()
    b_x_numpy = b_x_numpy.transpose(0,2,3,1)
    b_y_numpy = b_y.data.numpy()
    pre_lab_numpy = pre_lab.data.numpy()

    for ii in range(batchsize):
        b_x =inv_normalize_image(b_x_numpy[ii])
        b_y = label2image(b_y_numpy[ii])
        pre_lab = label2image(pre_lab_numpy[ii])

        # print('pre_lab:{} | b_y:{} |b_x:{}  '.format(pre_lab.shape, b_y.shape,b_x.shape))
        b_x = b_x.transpose((2, 0, 1))
        b_y = b_y.transpose((2, 0, 1))
        pre_lab = pre_lab.transpose((2, 0, 1))
        # b_x = torch.tensor(b_x)
        # b_y = torch.tensor(b_y)
        # pre_lab = torch.tensor(pre_lab)

        # print('pre_lab:{} | b_y:{} |b_x:{}  '.format(pre_lab.shape, b_y.shape,b_x.shape))
        # b_y = torch.unsqueeze(b_y, dim=1)
        # pre_lab = torch.unsqueeze(pre_lab, dim=1)

        # pre_lab = pre_lab.view(2, 1, 320, 480)
        viz = Visdom(env='step1')
        # b_x = b_x.astype(np.uint8)
        b_y = b_y.astype(np.uint8)
        pre_lab = pre_lab.astype(np.uint8)
        viz.image(b_x, win='b_x', opts={'title': 'b_x'})
        viz.image(b_y,  win='b_y', opts={'title': 'b_y'})
        viz.image(pre_lab, win='pre_lab', opts={'title': 'pre_lab'})

draw_pic(b_x,b_y,pre_lab)
print('draw _ finished')