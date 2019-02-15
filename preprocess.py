import h5py
import numpy as np
import os
from config import  config_dict
import  math
import utils

clip_min_max = [
        #### s1
        (-0.7, 0.7), (-0.5, 0.5),
        (-2.0, 2.0), (-2.0, 2.0),
        (0.0, 0.25), (0.0, 1.0),
        (-1.0, 1.0), (-0.2, 0.2),
        #### s2
        (0.0, 0.5), (0.0, 0.5), (0.0, 0.5),  ###BGR
        (0.0, 0.5), (0.0, 0.5), (0.0, 0.5), (0.0, 0.5), (0.0, 0.5), (0.0, 0.5), (0.0, 0.5),  ###这些波段有双峰现象
    ]
def clip_by_min_max(data_18, eas_scale=1):
    """
    :param data_18:  N *32*32*18 data
    :param eas_scale: 放宽系数
    :return:
    """
    print('clip data by channel specific min max val')
    assert len(data_18.shape) == 4
    assert data_18.shape[3]==18
    for i in range(18):
        min, max = clip_min_max[i]
        min, max = eas_scale*min, eas_scale*max
        data_18[:, :, :, i] = np.clip(data_18[:, :, :, i], min, max)
    return data_18

def simple_clip(data, min=-10,max=10):
    data = np.clip(data,min,max)
    return data

def data_norm_0_1(data_18):
    print('norm data by min max val')
    assert len(data_18.shape) == 4
    assert data_18.shape[3] == 18
    for i in range(18):
        min, max = clip_min_max[i]
        data_18[:, :, :, i] = (data_18[:, :, :, i]- min)/ (max-min)
    return data_18



def load_raw_data(path,load_label=True):
    fid = h5py.File(path,'r')
    s1    = fid['sen1']
    s2    = fid['sen2']
    if load_label:
        label = fid['label']
        return s1,s2,label
    return s1,s2,None

def gen_data(s1,s2, label,  data_save_path,
             label_save_path,clip_func=simple_clip, norm_func=data_norm_0_1):
    ## check
    assert len(s1) == len(s2)
    if label is not None:
        assert len(s1) == len(label)
    s1 = np.array(s1,np.float32)
    s2 = np.array(s2,np.float32)
    s  = np.concatenate((s1,s2),3)
    print('data shape: ',s.shape)
    s  = clip_func(s)
    ### 不统一做归一化， 统一归一化影响效果
    ###s  = norm_func(s)
    np.save(data_save_path,s)
    print('Saved in %s' % data_save_path)
    if label is not None:
        label = np.array(label, np.float32)
        np.save(label_save_path,label)
        print('Saved in %s' % label_save_path)

def calc_trainset_mean_std(s1,s2):
    print('calc trainset mean and std...')
    s1 = np.array(s1, np.float32)
    s2 = np.array(s2, np.float32)
    s = np.concatenate((s1, s2), 3)
    s = clip_by_min_max(s)
    print('data shape: ', s.shape)
    means = np.zeros((18, 1), np.float32)
    stds = np.zeros((18, 1), np.float32)
    for i in range(18):
        mean = s[:, :, :, i].mean()
        means[i] = mean
        std = s[:, :, :, i].std(ddof=1)
        stds[i] = std
        print('channel %d, mean: %3f, std: %3f' % (i,mean,std) )
    data_dir_root = config_dict['data_root_dir']
    save_path = os.path.join(data_dir_root, 'preprocess_dir', 'means.npy')
    np.save(save_path, means)
    save_path = os.path.join(data_dir_root, 'preprocess_dir', 'stds.npy')
    np.save(save_path, stds)
    print(means)
    print(stds)


def preprocess():
    data_dir_root = config_dict['data_root_dir']
    train_h5_file = os.path.join(data_dir_root, 'training.h5')
    val_h5_file = os.path.join(data_dir_root, 'validation.h5')
    #test_h5_file = os.path.join(data_dir_root, 'round1_test_a_20181109.h5')
    #test_h5_file = os.path.join(data_dir_root, 'round1_test_b_20190104.h5')
    test_h5_file=os.path.join(data_dir_root, 'round2_test_b_20190211.h5')
    s1_train, s2_train, label_train = load_raw_data(train_h5_file)
    s1_val, s2_val, label_val = load_raw_data(val_h5_file)
    s1_test, s2_test, _ = load_raw_data(test_h5_file, load_label=False)

    clip_func = clip_by_min_max
    norm_func = data_norm_0_1#data_norm_mean_std

    # calc_trainset_mean_std(s1_train,s2_train)

    print('preprocess train data...')
    gen_data(s1_train, s2_train, label_train,
             os.path.join(config_dict['preprocess_dir'], 'stage1_train_data.npy'),
             os.path.join(config_dict['preprocess_dir'], 'stage1_train_label.npy'),
             clip_func, norm_func)
    print('preprocess val data...')
    gen_data(s1_val, s2_val, label_val,
             os.path.join(config_dict['preprocess_dir'], 'stage1_val_data.npy'),
             os.path.join(config_dict['preprocess_dir'], 'stage1_val_label.npy'),
             clip_func, norm_func)
    print('preprocess test data...')
    gen_data(s1_test, s2_test, None,
             os.path.join(config_dict['preprocess_dir'], 'stage2_test_data_b.npy'),
             None,
             clip_func, norm_func)
    print('finished!')




if __name__=='__main__':
    preprocess()

