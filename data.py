import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import cv2


##data
class LCZData(Dataset):
    def __init__(self,
                 data,
                 label,
                 shuffle=True,
                 is_train=True,
                 img_size=128,
                 with_mixup=False,
                 label_smooth_lambda=0.1):
        super(LCZData, self).__init__()
        self.data = data
        self.label = label
        print('data shape: ', self.data.shape)
        self.inxs = list(range(0, self.data.shape[0], 1))
        self.with_mixup = with_mixup
        if with_mixup:
            self.mixup_inxs = list(range(0, self.data.shape[0], 1))
            random.shuffle(self.mixup_inxs)
        self.data_len = len(self.inxs)
        print('samples: ', self.data_len)
        if shuffle:
            random.shuffle(self.inxs)
        self.is_train = is_train
        self.img_size=img_size
        self.label_smooth_lambda=label_smooth_lambda

    def __len__(self):
        return len(self.inxs)

    def on_epoch_finish(self):
        random.shuffle(self.mixup_inxs)
        random.shuffle(self.inxs)

    def _flip(self, x, percent=0.5):
        if np.random.random() < percent:
            if np.random.random() < percent:
                for i in range(x.shape[0]):
                    x[i] = np.fliplr(x[i])
            if np.random.random() < percent:
                for i in range(x.shape[0]):
                    x[i] = np.flipud(x[i])
        return x

    def _norm_data(self, data):
        for c in range(data.shape[2]):
            if c < 8:
                continue
            max = np.max(data[:, :, c])
            min = np.min(data[:, :, c])
            data[:, :, c] = (data[:, :, c] - min) / (max - min + 1e-8)
        return data

    def _aug(self, data, percent=0.5, maxr=0.004):
        if np.random.random() < percent:
            max = np.max(data[:, :, -10:])
            min = np.min(data[:, :, -10:])
            c = 1 + maxr * random.randint(-100, 100)
            b = (max - min) * maxr * random.randint(-100, 100)
            data[:, :, -10:] = c * data[:, :, -10:] + b
        return data

    def __getitem__(self, i):
        inx = self.inxs[i]
        data = self.data[inx, :, :, :].copy()
        # data = self._norm_data(data)
        data = data.transpose(2, 0, 1)
        if self.with_mixup:
            mixup_inx = self.mixup_inxs[i]
            mix_data = self.data[mixup_inx, :, :, :].copy()
            mix_data = self._norm_data(mix_data)
            mix_data = mix_data.transpose(2, 0, 1)
            data = (data + mix_data) / 2
        if self.is_train:
            data = self._flip(data)
            data = self._aug(data)
        label = None
        if self.label is not None:
            label = self.label[inx]
            if self.is_train and self.label_smooth_lambda is not None:
                label = label * (1 - self.label_smooth_lambda) + \
                        (1 - label) * self.label_smooth_lambda / 16
            if self.with_mixup:
                mixup_inx = self.mixup_inxs[i]
                label = (label + self.label[mixup_inx]) / 2

        dataResize = np.zeros((18, self.img_size, self.img_size), np.float32)
        for i in range(18):
            dataResize[i, :, :] = cv2.resize(data[i, :, :],
                                             (self.img_size, self.img_size),
                                             interpolation=cv2.BORDER_REFLECT101)
        ## to tensor
        data = torch.from_numpy(dataResize)
        if label is not None:
            label = torch.from_numpy(label)
        else:
            label = torch.Tensor([0])
        return data, label


def get_test_loader(data_path, bsize=32,img_size=128, proc_n=4):
    data = np.load(data_path)
    dset = LCZData(data,
                   None,
                   shuffle=False,
                   is_train=False,
                   img_size=img_size
                  )
    dataloader = DataLoader(dset, batch_size=bsize, shuffle=False, num_workers=proc_n)
    return dataloader


if __name__ == '__main__':
    pass
