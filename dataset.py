import math
import random

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from VCM import VCM
from util import GenIdx, IdentitySampler

train_hdf_ = '../VCM-pose/VCM-POSE-HDF5-Train.hdf5'
test_hdf_ = '../VCM-pose/VCM-POSE-HDF5-Test.hdf5'
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class VideoDataset_train(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir, dataset_rgb, seq_len=12, sample='evenly', transform=None, index1=None, index2=None):
        if index2 is None:
            index2 = []
        if index1 is None:
            index1 = []
        self.dataset_ir = dataset_ir
        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.index1 = index1
        self.index2 = index2


        self.hdf5_file = h5py.File('%s' % train_hdf_, 'r')

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):

        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[self.index2[index]]
        num_ir = len(img_ir_paths)
        img_rgb_paths, pid_rgb, camid_rgb = self.dataset_rgb[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len  # 这个地方的S是指的seq_len

        sample_clip_ir = []
        frame_indices_ir = list(range(num_ir))
        if num_ir < S:
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num_ir / S)
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))
        sample_clip_ir = np.array(sample_clip_ir)

        sample_clip_rgb = []
        frame_indices_rgb = list(range(num_rgb))
        if num_rgb < S:
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                sample_clip_rgb.append(list(pool_rgb))
        else:
            inter_val_rgb = math.ceil(num_rgb / S)
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                sample_clip_rgb.append(list(pool_rgb))
        sample_clip_rgb = np.array(sample_clip_rgb)

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            imgs_ir_p = []
            for index in indices:
                index = int(index)

                img_path = img_ir_paths[index]
                img_key_p = img_path[13:].replace('/', '_').replace('.jpg', '.npy')

                img = read_image(img_path)
                img = np.array(img)

                img_p = self.hdf5_file[img_key_p][()]

                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
                imgs_ir_p.append(img_p)

            imgs_ir = torch.cat(imgs_ir, dim=0)
            imgs_ir_p = torch.stack(imgs_ir_p, dim=0)
            imgs_ir_p = torch.from_numpy(imgs_ir_p).float()

            frame_indices = range(num_rgb)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]

            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break

                indices.append(index)
            indices = np.array(indices)
            imgs_rgb = []
            imgs_rgb_p = []
            for index in indices:
                index = int(index)
                img_path = img_rgb_paths[index]
                img_key_p = img_path[20:].replace('/', '_')

                img = read_image(img_path)
                img = np.array(img)

                img_p = self.hdf5_file[img_key_p][()]

                if self.transform is not None:
                    img = self.transform(img)

                imgs_rgb.append(img)
                imgs_rgb_p.append(img_p)
            imgs_rgb = torch.cat(imgs_rgb, dim=0)
            imgs_rgb_p = torch.stack(imgs_rgb_p, dim=0)
            imgs_rgb_p = torch.from_numpy(imgs_rgb_p).float()

            return imgs_ir, imgs_ir_p, pid_ir, camid_ir, \
                imgs_rgb, imgs_rgb_p, pid_rgb, camid_rgb
        elif self.sample == 'video_train':
            idx1 = np.random.choice(sample_clip_ir.shape[1], sample_clip_ir.shape[0])
            number_ir = sample_clip_ir[np.arange(len(sample_clip_ir)), idx1]
            imgs_ir = []
            imgs_ir_p = []  # 添加！！！
            for index in number_ir:
                index = int(index)
                img_path = img_ir_paths[index]
                img_key_p = img_path[13:].replace('/', '_').replace('.jpg', '.npy')
                # 这个地方的13是为了去掉固定的前缀使其文件名回归原始状态便于检索hdf5文件
                # 举例来说：img_path = '../VCM/train/0063/ir/D2/121.jpg'，img_key_p = '0063_ir_D2_121.npy'
                # 必须使得文件名称变为0063_ir_D2_121.npy格式的检索，才会成功读取hdf5文件中的数据

                img = read_image(img_path)
                img = np.array(img)

                img_p = self.hdf5_file[img_key_p][()]

                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
                imgs_ir_p.append(img_p)
            imgs_ir = torch.cat(imgs_ir, dim=0)

            imgs_ir_p = np.stack(imgs_ir_p, axis=0)
            imgs_ir_p = torch.from_numpy(imgs_ir_p).float()
            ######################################################################################################################

            idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
            number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
            imgs_rgb = []
            imgs_rgb_p = []  # 添加！！！
            for index in number_rgb:
                index = int(index)
                img_path = img_rgb_paths[index]
                img_key_p = img_path[13:].replace('/', '_').replace('.jpg', '.npy')

                img = read_image(img_path)
                img = np.array(img)

                img_p = self.hdf5_file[img_key_p][()]

                if self.transform is not None:
                    img = self.transform(img)

                imgs_rgb.append(img)
                imgs_rgb_p.append(img_p)
            imgs_rgb = torch.cat(imgs_rgb, dim=0)
            imgs_rgb_p = np.stack(imgs_ir_p, axis=0)
            imgs_rgb_p = torch.from_numpy(imgs_rgb_p).float()
            return imgs_ir, imgs_ir_p, pid_ir, camid_ir, \
                imgs_rgb, imgs_rgb_p, pid_rgb, camid_rgb
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


class VideoDataset_test(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None):
        # 定义了数据集、序列长度、采样策略和数据变换等参数。
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform


        self.hdf5_file = h5py.File('%s' % test_hdf_, 'r')

    def __len__(self):
        # 返回数据集中样本的个数
        return len(self.dataset)

    def __getitem__(self, index):
        # 根据索引获取一个样本
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num))

        if num < S:
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S)
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        # 'video_test'：使用预先计算好的视频片段采样方案，从每个视频样本中抽取相应的样本。
        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0]
            imgs_ir = []
            imgs_ir_p = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                img_key_p = img_path[12:].replace('/', '_').replace('.jpg', '.npy')

                img = read_image(img_path)
                img = np.array(img)

                img_p = self.hdf5_file[img_key_p][()]

                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
                imgs_ir_p.append(img_p)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            imgs_ir_p = np.stack(imgs_ir_p, axis=0)
            imgs_ir_p = torch.from_numpy(imgs_ir_p).float()
            return imgs_ir, imgs_ir_p, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


if __name__ == '__main__':
    """
    下面的代码将使用VCM数据集测试VideoDataset_train和VideoDataset_test类
    """
    vcm = VCM()
    rgb_pos, ir_pos = GenIdx(vcm.rgb_label, vcm.ir_label)

    sampler = IdentitySampler(vcm.ir_label, vcm.rgb_label, rgb_pos, ir_pos, 2, 64)
    index1 = sampler.index1
    index2 = sampler.index2
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((288, 144)),
        transforms.Pad(10),

        transforms.RandomCrop((288, 144)),
        # T.Random2DTranslation(height, width),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pose_dataset = VideoDataset_train(vcm.train_ir, vcm.train_rgb, seq_len=12, sample='video_train', transform=transform_train,
                                      index1=index1, index2=index2)

    train_loader = torch.utils.data.DataLoader(
        pose_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    # TODO: 这里的num_workers=0，无法多线程读取hdf5文件。在之后的过程中需要将hdf5文件的打开放在__init__函数中，以便多线程读取。

    for batch_idx, (imgs_ir, imgs_ir_p, pids_ir, camids_ir, imgs_rgb, imgs_rgb_p, pids_rgb, camids_rgb) in enumerate(
            train_loader):
        print(imgs_ir.shape)
        print(imgs_ir_p.shape)
        print(pids_ir.shape)
        print('-------------------------------------------')
        print(imgs_rgb.shape)
        print(imgs_rgb_p.shape)
        print(pids_rgb.shape)
        break