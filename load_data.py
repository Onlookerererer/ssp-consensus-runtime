import os
import random
import numpy as np
import scipy.io as sio
import torch
import h5py
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class CustomDataSet(Dataset):
    def __init__(self, images, texts, img_labels, txt_labels, ori_labels):
        self.images = images
        self.texts = texts
        self.img_labels = img_labels
        self.txt_labels = txt_labels
        self.ori_labels = ori_labels

    def __getitem__(self, index):
        img = torch.from_numpy(self.images[index]).float() if isinstance(self.images[index], np.ndarray) else self.images[index].float()
        text = torch.from_numpy(self.texts[index]).float() if isinstance(self.texts[index], np.ndarray) else self.texts[index].float()
        img_label = torch.from_numpy(self.img_labels[index]).float() if isinstance(self.img_labels[index], np.ndarray) else self.img_labels[index].float()
        txt_label = torch.from_numpy(self.txt_labels[index]).float() if isinstance(self.txt_labels[index], np.ndarray) else self.txt_labels[index].float()
        ori_label = torch.from_numpy(self.ori_labels[index]).float() if isinstance(self.ori_labels[index], np.ndarray) else self.ori_labels[index].float()
        
        return img, text, img_label, txt_label, ori_label, torch.tensor(index, dtype=torch.long)

    def __len__(self):
        return len(self.images)

def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    ind = ind.ravel()
    
    if len(ind) == 0:
        print("Warning: Empty array input to ind2vec, returning empty array")
        return np.array([])
    
    if N is None:
        N = ind.max() + 1
    vec = np.zeros((len(ind), N), dtype='int16')
    vec[np.arange(len(ind)), ind] = 1
    return vec

# 重写：输入partial_length=每个样本候选标签总数（包含真实标签）
def get_partiallabels(labels, partial_length=2, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    if len(labels.shape) == 2:
        train_labels = np.argmax(labels, axis=1)
        K = labels.shape[1]
        n = labels.shape[0]
    else:
        train_labels = labels.copy()
        if np.min(train_labels) > 1:
            raise RuntimeError('testError: The minimum value of the label exceeds 1.')
        elif np.min(train_labels) == 1:
            train_labels = train_labels - 1
        K = int(np.max(train_labels) - np.min(train_labels) + 1)
        n = train_labels.shape[0]
    
    # 合法性校验
    if partial_length < 1:
        raise ValueError("partial_length must >= 1")
    if partial_length > K:
        raise ValueError(f"partial_length={partial_length} cannot exceed total classes K={K}")

    partial_labels = np.zeros((n, K), dtype=int)

    for j in range(n):
        true_cls = train_labels[j]
        # 所有其他类别
        other_cls = list(set(range(K)) - {true_cls})
        # 需要额外采样的数量
        sample_num = partial_length - 1
        # 随机选sample_num个其他类
        selected_others = np.random.choice(other_cls, size=sample_num, replace=False)
        # 真实标签+采样标签全部置1
        selected_all = [true_cls] + list(selected_others)
        partial_labels[j, selected_all] = 1
    
    return partial_labels

# 参数由partial_ratio改为partial_length
def get_loader(data_name, batch_size, partial_length):
    base_seed = 1
    np.random.seed(base_seed)
    random.seed(base_seed)
    torch.manual_seed(base_seed)

    img_train, text_train, label_train_img = None, None, None
    img_test, text_test, label_test_img = None, None, None
    img_valid, text_valid, label_valid_img = None, None, None

    if data_name == 'wiki':
        valid_len = 231
        path = 'datasets/wiki.mat'
        data = sio.loadmat(path)
        img_train = data['img_train']
        text_train = data['text_train']
        label_train_img = data['label_train'].reshape([-1,1]).astype('int16') 

        img_test = data['img_test']
        text_test = data['text_test']
        label_test_img = data['label_test'].reshape([-1,1]).astype('int16') 

        img_valid = img_test[0:valid_len]
        text_valid = text_test[0:valid_len]
        label_valid_img = label_test_img[0:valid_len]

        img_test = img_test[valid_len:]
        text_test = text_test[valid_len:]
        label_test_img = label_test_img[valid_len:]
    
    elif data_name == 'INRIA-Websearch':
        path = 'datasets/inria-websearch.mat'
        data = sio.loadmat(path)
        img_train = data['img_train'].astype('float32')
        text_train = data['text_train'].astype('float32')
        label_train_img = data['label_train'].reshape([-1,1]).astype('int16')

        img_valid = data['img_test'].astype('float32')
        text_valid = data['text_test'].astype('float32')
        label_valid_img = data['label_test'].reshape([-1,1]).astype('int16')

        img_test = data['img_test'].astype('float32')
        text_test = data['text_test'].astype('float32')
        label_test_img = data['label_test'].reshape([-1,1]).astype('int16')
        
    elif data_name == 'xmedianet':
        target_valid_len = 4000
        path = 'datasets/xmedianet.mat'
        all_data = sio.loadmat(path)
        img_test = all_data['img_test'].astype('float32')
        img_train = all_data['img_train'].astype('float32')
        text_test = all_data['text_test'].astype('float32')
        text_train = all_data['text_train'].astype('float32')

        label_test_img = all_data['label_test'].reshape([-1,1]).astype('int64')
        label_train_img = all_data['label_train'].reshape([-1,1]).astype('int64')

        test_total_len = len(img_test)
        valid_len = min(target_valid_len, int(test_total_len * 0.8))
        if valid_len == 0: 
            valid_len = int(len(img_train) * 0.1)
            img_valid = img_train[-valid_len:]
            text_valid = text_train[-valid_len:]
            label_valid_img = label_train_img[-valid_len:]
        else:
            img_valid = img_test[0:valid_len]
            text_valid = text_test[0:valid_len]
            label_valid_img = label_test_img[0:valid_len]

            img_test = img_test[valid_len:]
            text_test = text_test[valid_len:]
            label_test_img = label_test_img[valid_len:]

    elif data_name == 'nus-wide':
        valid_len = 5000
        path = 'datasets/nus_wide_deep_doc2vec_data_42941.h5py'

        with h5py.File(path, 'r') as h:
            img_train = h['train_imgs_deep'][()].astype('float32')
            text_train = h['train_texts'][()].astype('float32') if 'train_texts' in h else h['train_text'][()].astype('float32')
            label_train_idx = h['train_imgs_labels'][()].astype('int64')
            label_train_idx -= np.min(label_train_idx)
            label_train_img = label_train_idx.reshape([-1, 1])

            img_test = h['test_imgs_deep'][()].astype('float32')
            text_test = h['test_texts'][()].astype('float32') if 'test_texts' in h else h['test_text'][()].astype('float32')
            label_test_idx = h['test_imgs_labels'][()].astype('int64')
            label_test_idx -= np.min(label_test_idx)
            label_test_img = label_test_idx.reshape([-1, 1])

            valid_flag = True
            try:
                img_valid = h['valid_imgs_deep'][()].astype('float32')
                text_valid = h['valid_texts'][()].astype('float32') if 'valid_texts' in h else h['valid_text'][()].astype('float32')
                label_valid_idx = h['valid_imgs_labels'][()].astype('int64')
                label_valid_idx -= np.min(label_valid_idx)
                label_valid_img = label_valid_idx.reshape([-1, 1])
            except KeyError:
                valid_flag = False
                test_total_len = len(img_test)
                valid_len = min(valid_len, int(test_total_len * 0.8))
                img_valid = img_test[0:valid_len]
                text_valid = text_test[0:valid_len]
                label_valid_img = label_test_img[0:valid_len]
                img_test = img_test[valid_len:]
                text_test = text_test[valid_len:]
                label_test_img = label_test_img[valid_len:]
    else:
        raise ValueError(f"Error: Unsupported dataset '{data_name}'. Please choose from 'wiki', 'nus-wide', 'INRIA-Websearch', or 'xmedianet'.")

    img_train = img_train.astype('float32')
    img_valid = img_valid.astype('float32')
    img_test = img_test.astype('float32')
    text_train = text_train.astype('float32')
    text_valid = text_valid.astype('float32')
    text_test = text_test.astype('float32')
    
    label_train = label_train_img
    label_valid = label_valid_img
    label_test = label_test_img
    
    if len(label_train.shape) == 1 or label_train.shape[1] == 1:
        label_train = ind2vec(label_train.reshape([-1,1])).astype('int16') if len(label_train) > 0 else np.array([])
        label_valid = ind2vec(label_valid.reshape([-1,1])).astype('int16') if len(label_valid) > 0 else np.array([])
        label_test  = ind2vec(label_test.reshape([-1,1])).astype('int16') if len(label_test) > 0 else np.array([])
    
    root_dir = 'results/partial_labels'
    os.makedirs(root_dir, exist_ok=True)
    
    # 文件名替换为partial_length标识
    img_partial_file = os.path.join(root_dir, f'{data_name}_img_partial_labels_len_{partial_length}.mat')
    txt_partial_file = os.path.join(root_dir, f'{data_name}_txt_partial_labels_len_{partial_length}.mat')

    if os.path.exists(img_partial_file):
        img_label_partial = sio.loadmat(img_partial_file)['img_partial_label']
    else:
        # 传参改为partial_length
        img_label_partial = get_partiallabels(label_train, partial_length=partial_length, seed=base_seed+1)
        sio.savemat(img_partial_file, {'img_partial_label': img_label_partial})

    if os.path.exists(txt_partial_file):
        txt_label_partial = sio.loadmat(txt_partial_file)['txt_partial_label']
    else:
        txt_label_partial = get_partiallabels(label_train, partial_length=partial_length, seed=base_seed+2)
        sio.savemat(txt_partial_file, {'txt_partial_label': txt_label_partial})
        
    imgs = {'train': img_train, 'valid': img_valid}
    texts = {'train': text_train, 'valid': text_valid}
    img_labels = {'train': img_label_partial, 'valid': label_valid}
    txt_labels = {'train': txt_label_partial, 'valid': label_valid}
    ori_labels = {'train': label_train, 'valid': label_valid}
    
    dataset = {
        x: CustomDataSet(
            images=imgs[x],
            texts=texts[x],
            img_labels=img_labels[x],
            txt_labels=txt_labels[x],
            ori_labels=ori_labels[x]
        ) for x in ['train', 'valid'] if len(imgs[x]) > 0
    }

    shuffle = {'train': True, 'valid': False}
    dataloader = {
        x: DataLoader(
            dataset[x],
            batch_size=batch_size,
            shuffle=shuffle[x],
            num_workers=0, 
            pin_memory=True
        ) for x in ['train', 'valid'] if x in dataset
    }

    img_dim = img_train.shape[1] if len(img_train) > 0 else 0
    text_dim = text_train.shape[1] if len(text_train) > 0 else 0
    num_train = img_train.shape[0] if len(img_train) > 0 else 0
    num_class = label_train.shape[1] if len(label_train) > 0 else 0

    input_data_par = {
        'img_test': img_test,
        'text_test': text_test,
        'label_test': label_test,
        'img_valid': img_valid,
        'text_valid': text_valid,
        'label_valid': label_valid,
        'img_train': img_train,
        'text_train': text_train,
        'num_train': num_train,
        'label_train': label_train,
        'img_dim': img_dim,
        'text_dim': text_dim,
        'num_class': num_class,
        'img_partial_label': img_label_partial,
        'txt_partial_label': txt_label_partial
    }
    
    return dataloader, input_data_par