from yolox.models.yolox import YOLOX
import torch
import torch.nn as nn
import numpy as np

model = YOLOX()
# 加载稀疏训练的模型
model.load_state_dict(torch.load('D:/deeplearning/YOLOX-main/YOLOX_outputs/yolox_voc_s/yolox_s.pth'))
total = 0  # 统计所有BN层的参数量
for m in model.named_modules():
    if isinstance(m, nn.BatchNorm2d):
        # print(m.weight.data.shape[0])  # 每个BN层权重w参数量：64/128/256
        # print(m.weight.data)
        total += m.weight.data.shape[0]

print("所有BN层总weight数量：", total)

bn_data = torch.zeros(total)
index = 0
for m in model.named_modules():
    # 将各个BN层的参数值拷贝到bn中
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn_data[index:(index + size)] = m.weight.data.abs().clone()
        index = size
# 对bn中的weight值排序
data, id = torch.sort(bn_data)
percent = 0.7  # 保留70%的BN层通道数
thresh_index = int(total * percent)
thresh = data[thresh_index]  # 取bn排序后的第thresh_index索引值为bn权重的截断阈值

# 制作mask
pruned_num = 0  # 统计BN层剪枝通道数
cfg = []  # 统计保存通道数
cfg_mask = []  # BN层权重矩阵，剪枝的通道记为0，未剪枝通道记为1

for k, m in enumerate(model.named_modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        # print(weight_copy)
        mask = weight_copy.gt(thresh).float()  # 阈值分离权重
        # print(mask)
        # exit()
        pruned_num += mask.shape[0] - torch.sum(mask)  #
        # print(pruned_num)
        m.weight.data.mul_(mask)  # 更新BN层的权重，剪枝通道的权重值为0
        m.bias.data.mul_(mask)

        cfg.append(int(torch.sum(mask)))  # 记录未被剪枝的通道数量
        cfg_mask.append(mask.clone())
        print("layer index:{:d}\t total channel:{:d}\t remaining channel:{:d}".format(k, mask.shape[0],
                                                                                      int(torch.sum(mask))))
    elif isinstance(m, nn.AvgPool2d):
        cfg.append("A")

pruned_ratio = pruned_num / total
print("剪枝通道占比：", pruned_ratio)
print(cfg)
newmodel = YOLOX(cfg)
# print(newmodel)
# from torchsummary import summary
# print(summary(newmodel,(3,20,20),1))

layer_id_in_cfg = 0  # 层
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]  # 第一个BN层对应的mask
# print(cfg_mask)
# print(end_mask)

for (m0, m1) in zip(model.named_modules(), newmodel.named_modules()):  # 以最少的为准
    if isinstance(m0, nn.BatchNorm2d):
        # idx1=np.squeeze(np.argwhere(np.asarray(end_mask.numpy())))#获得mask中非零索引即未被减掉的序号
        # print(idx1)
        # exit()
        # idx1=np.array([1])
        # # print(idx1)
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
            # print(idx1)
        # exit()
        # 将旧模型的参数值拷贝到新模型中
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()

        layer_id_in_cfg += 1  # 下一个mask
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):  # 输入
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.numpy())))  # 输入非0索引
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.numpy())))  # 输出非0索引

        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))

        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.numpy())))  # 输入非0索引
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))

        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save(newmodel.state_dict(), 'D:/deeplearning/YOLOX-main/YOLOX_outputs/yolox_voc_s/prune_yolox.pth')

