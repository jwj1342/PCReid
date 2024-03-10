import time
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from VCM import VCM
from config.config import ExperimentConfig
from dataset import VideoDataset_train, VideoDataset_test
from feature_net import embed_net as FeatureNet
from loss import OriTripletLoss
from test import test_general
from transforms import transform_train, transform_test
from util import IdentitySampler, GenIdx, load_model, save_model
import wandb

data_set = VCM()  # 从这个地方引用了数据集
configs = ExperimentConfig().get_config()  # 导入配置文件
seq_length = configs['Training_Settings']['seq_len']  # 从配置文件中获取序列长度
batch_size = configs['Training_Settings']['batch_size']  # 从配置文件中获取batch大小
num_pos = configs['Model_Settings']['num_pos']  # 从配置文件中获取num_pos
workers = configs['Training_Settings']['workers']  # 从配置文件中获取workers
margin = configs['Model_Settings']['margin']  # 从配置文件中获取margin
test_batch = configs['Training_Settings']['test_batch']  # 从配置文件中获取test_batch
lr = configs['Training_Settings']['lr']  # 从配置文件中获取学习率
epochs = configs['Training_Settings']['epochs']  # 从配置文件中获取epoch
loader_batch = batch_size * num_pos

query_loader = DataLoader(
    VideoDataset_test(data_set.query, seq_len=seq_length, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=workers)

gallery_loader = DataLoader(
    VideoDataset_test(data_set.gallery, seq_len=seq_length, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=workers)
query_loader_1 = DataLoader(
    VideoDataset_test(data_set.query_1, seq_len=seq_length, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=workers)

gallery_loader_1 = DataLoader(
    VideoDataset_test(data_set.gallery_1, seq_len=seq_length, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=workers)

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
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=1e-3, weight_decay=5e-4)  # 优化器

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

wandb.login()
wandb.init(
    project="PoseReid",
    config=configs,
    notes="初步融合原本模态与pose模态的训练过程",
)
best_mAP = 0
# load_model(net, "best_mAp_sum_0.29.pth")
wandb.watch(net)
for epoch in range(epochs):
    net.train()
    start_time = time.time()
    accumulated_loss = 0.0  # 初始化累计损失
    for batch_idx, (imgs_ir, imgs_ir_p, pid_ir, camid_ir, imgs_rgb, imgs_rgb_p, pid_rgb, camid_rgb) in enumerate(
            train_loader):
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
        accumulated_loss += loss.item()  # 累计损失可视化用
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0 and batch_idx > 0:  # 确保batch_idx为20的倍数且不为0时才进行打印
            end_time = time.time()  # 记录当前时间
            elapsed_time = end_time - start_time  # 计算经过的时间
            avg_loss = accumulated_loss / 20  # 计算平均损失
            print('Epoch: [{}][{}/{}]\t'
                  'Avg Loss {:.4f} Loss_id {:.4f} Loss_tri {:.4f} Time {:.2f} seconds'.format(
                epoch, batch_idx, len(train_loader), avg_loss, loss_id.item(), loss_tri.item(), elapsed_time))
            wandb.log({
                'Epoch': epoch,
                'Batch_idx': batch_idx,
                'Avg Loss': avg_loss,
                'Loss_id': loss_id.item(),
                'Loss_tri': loss_tri.item(),
                'Time': elapsed_time
            })
            accumulated_loss = 0.0  # 重置累计损失为0，为下一个20个批次做准备
            start_time = time.time()

    if epoch %5==0:
        net.eval()
        cmc_t2v, mAP_t2v = test_general(gallery_loader, query_loader, net, ngall, nquery, modal=1)
        wandb.log({"epoch": epoch, "mAP_t2v": mAP_t2v})
        wandb.log({"epoch": epoch, "t2v-Rank-1": cmc_t2v[0]})
        wandb.log({"epoch": epoch, "t2v-Rank-20": cmc_t2v[4]})

        cmc_v2t, mAP_v2t = test_general(gallery_loader_1, query_loader_1, net, ngall_1, nquery_1, modal=2)
        wandb.log({"epoch": epoch, "mAP_v2t": mAP_v2t})
        wandb.log({"epoch": epoch, "v2t-Rank-1": cmc_v2t[0]})
        wandb.log({"epoch": epoch, "v2t-Rank-20": cmc_v2t[4]})

        if mAP_t2v + mAP_v2t > best_mAP:
            best_mAP = mAP_t2v + mAP_v2t
            save_model(net, f"best_mAp_sum_pure_redoBase_0.005_{best_mAP}.pth")
