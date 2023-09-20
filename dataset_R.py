import numpy as np
import torch
from torchsummary import summary
from torchvision import transforms
from torch.utils import data
from PIL import Image
from torchvision.models import vgg19
import torch.nn.functional as F
from torch import nn, optim
# 读取文档,识别要用的数据
# 由于VOC2012数据集并不都用于语义分割，因此一万多张图片里有一些数据是我们需要舍弃的。
# 数据集里的\ImageSets\Segmentation里的txt文档描述了哪些数据可以用作语义分割：
def read_image_path(root):
    image = np.loadtxt(root, dtype = str)
    n = len(image) # 数据集尺寸
    data, label = [None]*n, [None]*n
    for i, fname in enumerate(image):
        data[i] = 'E:/deeplearning/dataB/VOCtrainval_11-May-2012_B/VOCdevkit/VOC2012/JPEGImages/%s.jpg' %(fname)        # 数据集
        label[i] = 'E:/deeplearning/dataB/VOCtrainval_11-May-2012_B/VOCdevkit/VOC2012/SegmentationClass/%s.png'%(fname) # 标签
    return data, label

# 由于语义分割是像素级别的分类，因此标签和原图必须完美的匹配
# 这时候如果使用transforms模块自带的数据增强方法，因为是随机方法，因此处理后就会导致图像和标签在像素上不匹配的情况，
# 因此我们自定义图像增强方法:
# 由于是全卷积网络，图像的大小固不固定无所谓
def rand_crop(data, label, high, width):  # high, width为裁剪后图像的固定宽高(320x480)
    im_width, im_high = data.size
    # 生成随机点位置
    left = np.random.randint(0, im_width - width)
    top = np.random.randint(0, im_high - high)
    right = left + width
    bottom = top + high
    # 图像随机裁剪(图像和标签一一对应)
    data = data.crop((left, top, right, bottom))
    label = label.crop((left, top, right, bottom))

    # 图像随机翻转(图像和标签一一对应)
    angle = np.random.randint(-15, 15)
    data = data.rotate(angle)  # 逆时针旋转
    label = label.rotate(angle)  # 逆时针旋转
    return data, label


# 预处理
def img_transforms(data, label, high, width):
    data, label = rand_crop(data, label, high, width)
    data_tfs = transforms.Compose([
        transforms.ToTensor(),
        # 标准化，据说这6个参数是在ImageNet上百万张数据里提炼出来的，效果最好
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data = data_tfs(data)

    label = torch.from_numpy(np.array(label))
    # 去除边缘标签！！！！
    label_without_border = torch.where(label < 255, label, torch.tensor([0], dtype=torch.uint8))

    return data, label_without_border

#自定义数据集：
class MyDataset(data.Dataset):
    def __init__(self, data_root, high, width):
        self.data_root = data_root
        self.high = high
        self.width = width
        self.imtransform = img_transforms
        data_list, label_list = read_image_path(root = data_root)
        self.data_list = self.filter(data_list)
        self.label_list = self.filter(label_list)

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label) #.convert('RGB')
        img, label = self.imtransform(img, label, self.high, self.width)
        return img, label

    def __len__(self):
        return len(self.data_list)

    # 防止rand_crop函数越界报错,过滤掉图像尺寸小于high，width 的图像
    def filter(self, images):
        return [im for im in images if (Image.open(im).size[1] > self.high and Image.open(im).size[0] > self.width)]



#
# high, width = 320, 480
# BATCHSIZE = 2
# voc_train = MyDataset(
#     "E:/deeplearning/dataB/VOCtrainval_11-May-2012_B/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt", high, width)
# voc_val = MyDataset("E:/deeplearning/dataB/VOCtrainval_11-May-2012_B/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
#                     high, width)
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train_loader = data.DataLoader(voc_train, batch_size=BATCHSIZE, shuffle=True)
# val_loader = data.DataLoader(voc_val, batch_size=BATCHSIZE, shuffle=True)
# print('训练集大小:{}'.format(voc_train.__len__()))
# print('验证集大小:{}'.format(voc_val.__len__()))

