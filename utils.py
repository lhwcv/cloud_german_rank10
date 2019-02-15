import os
import numpy as np
import torch
import errno
EPS = 1e-7


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)

def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]


def get_ensemble_models_path(root_dir):
    def get_score(dir):
        try:
            with open(dir + '/log.txt') as f:
                lines = f.readlines()
                score = float(lines[-1][-7:-2])
                return score
        except:
            return 0.0
    sub_dirs = [os.path.join(root_dir,p) for p in os.listdir(root_dir) if
                os.path.isdir(os.path.join(root_dir,p))]
    print(sub_dirs)
    ensemble_models_path=[]
    for subdir in sub_dirs:
        for p in os.listdir(subdir):
                ensemble_models_path.append(subdir+'/'+p)
    score_dict = {}
    for p in ensemble_models_path:
        score_dict[p] = get_score(p)
    score_avg=0
    print(len(ensemble_models_path))
    for p in ensemble_models_path:
        print(p, score_dict[p])
        score_avg+=score_dict[p]
    print('*****************************************')
    print('local score avg: ',score_avg/len(ensemble_models_path))
    print('*****************************************')
    return  ensemble_models_path