import random
import numpy as np
from colorcet import cm
from matplotlib import pyplot as plt
import torch
from torch.utils import data
import torch.nn.functional as F
from FCNtest.FCN8S_R import FCN8s
from FCNtest.dataset_R import MyDataset


from dataset_R import img_transforms
# img = 'E:/deeplearning/dataB/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/%s.jpg' %(fname)        # 数据集
# label = 'E:/deeplearning/dataB/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass/%s.png'%(fname) # 标签
# high, width = 320, 480
# img = Image.open(img)
# label = Image.open(label) #.convert('RGB')
# img, label = img_transforms(img, label, high, width)



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



high, width = 320, 480
BATCHSIZE = 8
voc_train = MyDataset(
    "E:/deeplearning/dataB/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt", high, width)
voc_val = MyDataset("E:/deeplearning/dataB/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
                    high, width)
train_loader = data.DataLoader(voc_train, batch_size=BATCHSIZE, shuffle=True)
val_loader = data.DataLoader(voc_val, batch_size=BATCHSIZE, shuffle=True)

fcn8s = FCN8s(21).cpu()
fcn8s.load_state_dict(torch.load('fcn8s.pkl'))

for step ,(b_x, b_y) in enumerate(val_loader):
    if step > 0:
        break
    fcn8s.eval()
    b_x = b_x.float()
    b_y = b_y.long()
    out = fcn8s(b_x) # out:(BATCHSIZE, LabelNum, 320, 480)
    out = F.log_softmax(out, dim = 1)
    pre_lab = torch.argmax(out, 1)


    #可视化一个batch图像：
    b_x_numpy = b_x.data.numpy()
    b_x_numpy = b_x_numpy.transpose(0,2,3,1)
    b_y_numpy = b_y.data.numpy()
    pre_lab_numpy = pre_lab.data.numpy()

    plt.figure(figsize = (16, 5))
    for ii in range(BATCHSIZE):
        plt.subplot(3,BATCHSIZE,ii+1)
        plt.imshow(inv_normalize_image(b_x_numpy[ii]))
        plt.axis('off')

        plt.subplot(3,BATCHSIZE,ii+9)
        plt.imshow(label2image(b_y_numpy[ii]))
        plt.axis('off')

        plt.subplot(3,BATCHSIZE,ii+17)
        plt.imshow(label2image(pre_lab_numpy[ii]))
        plt.axis('off')
        high, width = 320, 480
        print(np.sum(pre_lab_numpy[ii] == b_y_numpy[ii])/(high * width))

    plt.subplots_adjust(wspace = 0.01, hspace = 0.01)
    plt.show()
