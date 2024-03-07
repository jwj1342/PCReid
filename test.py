import numpy as np
import torch
from torch.autograd import Variable

from util import evaluate


def extract_features_no_grad(data_loader, feature_dimension, net, modal=0):
    """提取特征的通用函数"""
    features = np.zeros((len(data_loader.dataset), feature_dimension))
    pids = []
    camids = []
    ptr = 0
    with torch.no_grad():
        for batch_idx, (img, imgs_ir_p, pid, camid) in enumerate(data_loader):
            input_imgs = Variable(img.float().cuda())
            batch_num = input_imgs.size(0)
            feats_cls = net(input_imgs, input_imgs, input_imgs, input_imgs, modal=modal)
            features[ptr:ptr + batch_num, :] = feats_cls.detach().cpu().numpy()
            ptr += batch_num
            pids.extend(pid)
            camids.extend(camid)
    return features, np.asarray(pids), np.asarray(camids)

def test_general(gallery_loader, query_loader, net, ngall, nquery, modal=0):
    """通用测试函数, 用于测试单个模型的性能"""
    net.eval()

    print('Extracting Gallery Feature...')
    gall_feat, g_pids, g_camids = extract_features_no_grad(gallery_loader, 2048, net, modal=modal)

    print('Extracting Query Feature...')
    query_feat, q_pids, q_camids = extract_features_no_grad(query_loader, 2048, net, modal=modal)

    # 计算相似度
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)

    ranks = [1, 5, 10, 20]
    print("Results ----------")
    print("testmAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc, mAP