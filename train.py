import numpy as np
import torch
import utils
import timeit
import os
from Nadam import Nadam
from torch.utils.data import DataLoader,Dataset
from torch.optim import lr_scheduler
import tqdm
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold,KFold
from data import  LCZData
import random
from model import *

def eval(model, dataloader):
    rightN=0
    for data,label in dataloader:
        data = data.cuda()
        pred = model(data)
        pred = np.argmax(pred.data.cpu().numpy(),1)
        gt = label.numpy()
        gt = np.argmax(gt, 1)
        rightN+=(pred==gt).sum()
    acc = rightN/len(dataloader.dataset)
    print('acc:',acc)
    return acc


def train(model, train_loader, val_loader, output_dir,
          cfg,
          ):
    num_epochs   = cfg['epoches']
    base_lr      = cfg['baselr']
    early_stop_n = cfg['early_stop_n'],
    utils.create_dir(output_dir)

    optim = Nadam(model.parameters(), lr=base_lr)
    scheduler = lr_scheduler.MultiStepLR(optim, [6, 12, 18], gamma=0.5)
    logger = utils.Logger(os.path.join(output_dir, 'log.txt'))
    best_eval_score_avg = 0.0
    print('Begin training......')
    early_stop_counter = 0
    for epoch in range(num_epochs):
        scheduler.step(epoch)
        total_loss = 0
        rightN = 0
        if cfg['with_mixup'] and epoch < cfg['mixup_max_n']:
            if train_loader.dataset.with_mixup == True:
                train_loader.dataset.with_mixup = False
                print('Close mixup..')
            else:
                train_loader.dataset.with_mixup = True
                print('Open mixup..')

        for data, label in tqdm.tqdm(train_loader):
            optim.zero_grad()
            data = data.cuda()
            pred = model(data)
            label = label.cuda()
            loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
            loss = loss.sum(1)
            loss, _ = loss.topk(k=int(loss.size(0) * 0.9))
            loss = loss.mean()
            pred = np.argmax(pred.data.cpu().numpy(), 1)
            gt = label.cpu().numpy()
            gt = np.argmax(gt, 1)
            rightN += (pred == gt).sum()
            loss.backward()
            optim.step()
            total_loss += loss.item() * label.size(0)
        NN = len(train_loader.dataset)
        total_loss /= NN
        logger.write('epoch %d, \ttrain_loss: %.3f,'
                     ' train_score: %.3f' % (epoch, total_loss, rightN / NN))
        train_loader.dataset.on_epoch_finish()
        if epoch >= cfg['val_after_epoch']:
            model.train(False)
            eval_score = eval(model, val_loader)
            model.train(True)
            if eval_score >= best_eval_score_avg:
                early_stop_counter = 0
                model_path = os.path.join(output_dir, 'model_best.pth')
                torch.save(model.state_dict(), model_path)
                best_eval_score_avg = eval_score
            else:
                early_stop_counter += 1
            logger.write('epoch %d,' % (epoch) +
                         '\teval score: %.2f ' % (100 * eval_score) +
                         '( best:  %.2f)' % (100 * best_eval_score_avg))
            if early_stop_counter > early_stop_n:
                break
    print('**************************************************')

def main(cfg):
    all_data = np.load(cfg['preprocess_dir'] + '/stage1_train_data.npy')
    all_data = np.concatenate([all_data, np.load(cfg['preprocess_dir'] + '/stage1_val_data.npy')], 0)
    all_label = np.load(cfg['preprocess_dir'] + '/stage1_train_label.npy')
    all_label = np.concatenate([all_label, np.load(cfg['preprocess_dir'] + '/stage1_val_label.npy')], 0)
    folds = KFold(5, shuffle=True, random_state=666)
    for n_fold, (train_idx, val_idx) in enumerate(folds.split(all_label)):
        train_data = all_data[train_idx]
        val_data = all_data[val_idx]
        train_label = all_label[train_idx]
        val_label = all_label[val_idx]

        train_dset = LCZData(train_data, train_label, shuffle=True, is_train=True)
        val_dset = LCZData(val_data, val_label, shuffle=False, is_train=False)
        train_dataloader = DataLoader(train_dset, batch_size=cfg['batch_size'],
                                      shuffle=True, pin_memory=True, num_workers=6)
        val_dataloader = DataLoader(val_dset, batch_size=cfg['batch_size'],
                                    shuffle=False, pin_memory=True, num_workers=4)

        output_dir = cfg['model_outdir']
        output_dir = output_dir + '/model_0/fold_%d' % n_fold
        seed = random.randint(0, 100000)
        print('outdir:', output_dir)
        torch.manual_seed(seed)
        model=None
        if cfg['backbone']=='se50':
            model = GermanNetSE50().cuda()
        elif cfg['backbone']=='xcep':
            model = GermanNetXcep().cuda()
        else:
            model = GermanNetIncepRes().cuda()
        train(model, train_dataloader, val_dataloader, output_dir,cfg)


from  config import  config_dict
if __name__=='__main__':
    main(config_dict)