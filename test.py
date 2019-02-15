import torch
import numpy as np
from config import  config_dict
import os
from data import  get_test_loader
from model import  *
import tqdm
import datetime
import utils


def online_test(data_path,model_dir_list,
                loggits_save_path,
                test_img_size=128):
    models = []
    device = torch.device("cuda:0")
    for inx, i in enumerate(model_dir_list):
        if 'se50' in i:
            model=GermanNetSE50().to(device)
        elif 'xcep' in i:
            model=GermanNetXcep().to(device)
        else:
            model=GermanNetIncepRes().to(device)
        model.eval()
        model_path = os.path.join(i, 'model_best.pth')
        print(model_path)
        model.load_state_dict(torch.load(model_path))
        models.append(model)

    dataloader = get_test_loader(data_path,bsize=32,img_size=test_img_size)
    pred_npy = np.zeros((len(dataloader.dataset),17),np.float32)
    print('online test predicting.....')
    utils.create_dir(config_dict['commit_outdir'])
    submit_csv = config_dict['commit_outdir'] + \
                 datetime.datetime.now().strftime('%Y%m%d_%H%M%S') +  "_submit.csv"
    fout = open(submit_csv,'w')
    inx=0
    for data,label in tqdm.tqdm(dataloader):
        data=data.to(device)
        pred = models[0](data)
        for i in range(1, len(models)):
            pred_sub = models[i](data)
            pred += pred_sub
        pred = pred.data.cpu().numpy()
        pred_npy[inx:inx+pred.shape[0],:]=pred
        inx+=pred.shape[0]
        pred = np.argmax(pred, 1)
        
        for i in range(pred.shape[0]):
            one_hot = [0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0]
            one_hot[pred[i]]=1
            for j in  range(16):
                fout.write(str(one_hot[j])+',')
            fout.write(str(one_hot[16]) + '\n')
    fout.close()
    np.save(loggits_save_path,pred_npy)
    print('pred logits saved in: '+loggits_save_path)
    print('submit csv saved in: '+submit_csv)

if __name__=='__main__':
    model_path_list = []
    model_path_list.extend(utils.get_ensemble_models_path('../saved_models_se50_128size/'))
    data_path=config_dict['preprocess_dir'] + '/stage2_test_data_b.npy'
    utils.create_dir('./pred')
    loggits_save_path='pred/se50_128.npy'
    online_test(data_path,model_path_list,loggits_save_path)