import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from VCM import VCM
from config.config import ExperimentConfig
from dataset import VideoDataset_train
from feature_net import embed_net as FeatureNet
from loss import OriTripletLoss
from transforms import transform_train
from util import IdentitySampler, GenIdx

data_set = VCM()  # 从这个地方引用了数据集
configs = ExperimentConfig().get_config()  # 导入配置文件
seq_length = configs['Training_Settings']['seq_len']  # 从配置文件中获取序列长度
batch_size = configs['Training_Settings']['batch_size']  # 从配置文件中获取batch大小
num_pos = configs['Model_Settings']['num_pos']  # 从配置文件中获取num_pos
workers = configs['Training_Settings']['workers']  # 从配置文件中获取workers
margin = configs['Model_Settings']['margin']  # 从配置文件中获取margin
test_batch = configs['Training_Settings']['test_batch']  # 从配置文件中获取test_batch
loader_batch = batch_size * num_pos


nquery_1 = data_set.num_query_tracklets_1  # 测试集中红外图像数据的tracklet数：5099
ngall_1 = data_set.num_gallery_tracklets_1  # gallery中红外图像数据的tracklet数：4584

n_class = data_set.num_train_pids  # 训练集中的行人ID数：
nquery = data_set.num_query_tracklets  # 表示测试集中可见光图像数据的tracklet数：4584
ngall = data_set.num_gallery_tracklets  # gallery中可见光图像数据的tracklet数：5099

net = FeatureNet(low_dim=2048, class_num=702, drop=0.2, part=3, alpha=0.2, nheads=4, arch='resnet50', wpa=False)
net.to('cuda')  # 将网络发送到GPU上

criterion_CrossEntropy = nn.CrossEntropyLoss()  # 交叉熵损失函数
criterion_Triplet = OriTripletLoss(batch_size=loader_batch, margin=margin)  # 三元组损失函数
criterion_Triplet.to('cuda')  # 将三元组损失函数发送到GPU上
criterion_CrossEntropy.to('cuda')  # 将交叉熵损失函数发送到GPU上

# optimizer = torch.optim.Adam(net.parameters(), lr=0.0003, weight_decay=5e-4)  # 优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)  # 优化器

rgb_pos, ir_pos = GenIdx(data_set.rgb_label, data_set.ir_label)
sampler = IdentitySampler(data_set.ir_label, data_set.rgb_label, rgb_pos, ir_pos, num_pos, batch_size)
index1 = sampler.index1
index2 = sampler.index2

train_loader = DataLoader(  # 从这个地方引用了数据加载器
        VideoDataset_train(data_set.train_ir, data_set.train_rgb, seq_len=seq_length, sample='video_train',
                           transform=transform_train, index1=index1, index2=index2),
        sampler=sampler,
        batch_size=loader_batch, num_workers=workers,
        drop_last=True,
    )
# 提取数据
saved_batches = []
for batch_idx, data in enumerate(train_loader):
    if batch_idx < 5:  # 只保存前2个批次的数据
        saved_batches.append(data)
    else:
        break

num_epochs = 500  # 重复训练的轮数

for epoch in range(num_epochs):
    total_loss = 0
    for data in tqdm(saved_batches):
        imgs_ir, imgs_ir_p, pid_ir, camid_ir, imgs_rgb, imgs_rgb_p, pid_rgb, camid_rgb = data
        input1 = imgs_rgb
        input3 = imgs_rgb_p
        input2 = imgs_ir
        input4 = imgs_ir_p

        label1 = pid_rgb
        label2 = pid_ir
        labels = torch.cat((label1, label2), 0)  # 将两个模态的标签拼接起来，形成一个新的标签
        labels = Variable(labels.cuda())

        # 把两个模态的图片发送到GPU上
        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())
        input3 = input3.cuda()
        input4 = input4.cuda()

        # 将数据放入网络中进行训练，导出LSTM的最后时间步输出和分类头的输出
        output_pool, output_cls = net(input1, input2, input3, input4, seq_len=12)

        loss_id = criterion_CrossEntropy(output_cls, labels)  # 计算交叉熵损失
        loss_tri, _ = criterion_Triplet(output_pool, labels)  # 计算三元组损失

        loss = loss_id + loss_tri  # 总损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(saved_batches)}")
