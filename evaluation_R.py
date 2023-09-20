import numpy as np
import torch


def miou(predict, label, class_num=3):
    """
        计算多个batch的moiu
        @param predict: 模型预测值，Shape:[B, W, H]
        @param label: 标签，Shape:[B, W, H]
    """
    batch = label.shape[0]

    predict, label = predict.flatten(), label.flatten()

    # 忽略背景的话就 >0
    k = (predict >= 0) & (predict < class_num)

    # 计算混淆矩阵
    hist = torch.bincount(class_num * predict[k].type(torch.int32) + label[k], minlength=batch * (class_num ** 2)).reshape(batch, class_num, class_num)

    # 将多个batch合并为一个，如果传入的参数没有batch这个维度，可以注释掉这句话
    hist = hist.sum(0)

    # 计算各个类的iou
    miou = torch.diag(hist) / torch.maximum((hist.sum(1) + hist.sum(0) - torch.diag(hist)), torch.tensor(1))

    # 计算平均值miou
    return miou.mean()

def Iou(target_all, pred_all, n_class):
    """
    计算评价指标
    target是真实标签，shape为(h,w)，像素值为0，1，2...
    pred是预测结果，shape为(h,w)，像素值为0，1，2...
    n_class:为预测类别数量
    """
    pred_all = pred_all.to('cpu')
    target_all = target_all.to('cpu')
    iou_rec = []
    iou =[]
    for i in range(target_all.shape[0]):
        pred = pred_all[i]
        target = target_all[i]

        h, w = target.shape
        # 转为one-hot，shape变为(h,w,n_class)
        target_one_hot = np.eye(n_class)[target]
        pred_one_hot = np.eye(n_class)[pred]

        target_one_hot[target_one_hot != 0] = 1
        pred_one_hot[pred_one_hot != 0] = 1
        join_result = target_one_hot * pred_one_hot

        join_sum = np.sum(np.where(join_result == 1))  # 计算相交的像素数量
        pred_sum = np.sum(np.where(pred_one_hot == 1))  # 计算预测结果非0得像素数
        target_sum = np.sum(np.where(target_one_hot == 1))  # 计算真实标签的非0得像素数
        iou.append(join_sum / (pred_sum + target_sum - join_sum + 1e-6))
        iou_rec += iou
        # iou_rec.append(join_sum / (pred_sum + target_sum - join_sum + 1e-6))
    return iou_rec  #  iou是个list

def epoch_miou(eiou,iou):
    # eiou = eiou.to('cpu')
    # iou = iou.to('cpu')
    eiou += iou
    return eiou

def mean_iou(eiou):
    # eiou =eiou.to('cpu')
    ans = np.mean(eiou)
    return ans



def add_hist( juzhen,hist):
    juzhen = juzhen.to('cpu')
    hist = hist.to('cpu')
    juzhen = juzhen.add(hist)
    return juzhen

def miou_cul(juzhen):
    miou = torch.diag(juzhen) / torch.maximum((juzhen.sum(1) + juzhen.sum(0) - torch.diag(juzhen)), torch.tensor(1))
    return miou.mean()
# def mean(list):
#     list = list.to('cpu')
#     mean = np.mean(list)
#     return mean