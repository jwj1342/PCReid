import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Sampler


def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos

def extract_features_no_grad(data_loader, feature_dimension, net):
    """提取特征的通用函数"""
    features = np.zeros((len(data_loader.dataset), feature_dimension))
    pids = []
    camids = []
    ptr = 0
    with torch.no_grad():
        for batch_idx, (imgs, pids_batch, camids_batch) in enumerate(data_loader):
            input_imgs = Variable(imgs.float().cuda())
            batch_num = input_imgs.size(0)
            feats = net(input_imgs, input_imgs)
            features[ptr:ptr + batch_num, :] = feats.detach().cpu().numpy()
            ptr += batch_num
            pids.extend(pids_batch)
            camids.extend(camids_batch)
    return features, np.asarray(pids), np.asarray(camids)

class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
    """

    def __init__(self, train_thermal_label, train_color_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label = np.unique(train_color_label)
        # print('uni_label')
        # print(uni_label)
        self.n_classes = len(uni_label)

        N = np.minimum(len(train_color_label), len(train_thermal_label))
        for j in range(int(N / (batchSize * num_pos))):
            # print('aaa')
            # print(int(N/(batchSize*num_pos)))
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            batch_idx = list(batch_idx)
            for i in range(batchSize):
                # print('i')
                # print(i)
                # print(batch_idx)
                # print(len(batch_idx))
                # print(len(color_pos))
                # print(len(thermal_pos))
                # print(uni_label[-1])
                sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                    # index = np.hstack((index1, index2))
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
                    # index = np.hstack((index, sample_color))
                    # index = np.hstack((index, sample_thermal))
        self.N = int(N / (batchSize * num_pos)) * (batchSize * num_pos)
        print("index1.shape = {}".format(index1.shape))
        print("index2.shape = {}".format(index2.shape))
        # print("index.shape = {}".format(index.shape))
        print("index1 = {}".format(index1))
        print("index2 = {}".format(index2))
        # print("index = {}".format(index))
        print("N = {}".format(self.N))
        print("return iter {}".format(np.arange(len(index1))))
        self.index1 = index1
        self.index2 = index2
        # self.index = list(index)
        # print("index = {}".format(index))
        # self.N  = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))
        # return iter(self.index2)

    def __len__(self):
        return self.N