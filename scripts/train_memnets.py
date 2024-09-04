import argparse
import pprint
import os
import glob
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from memnets.utils import set_random_seed
from memnets.gme.model import MEMNet, MEMLayer, MEMEstimator
from memnets.processing.dataprocessing import Preprocessing

parser = argparse.ArgumentParser(description='Training with MEMnets')

parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--device', default='cpu', type=str, help='train the model with gpu or cpu')

parser.add_argument('--lagtimes', nargs='+', type=int, help='the encoder lag times to create time series: [\delta t, n_1 \delta t, ... , n_k \delta t]', required=True)

parser.add_argument('--encoder_sizes', nargs='+', type=int, help='the size of each layer in MEMnets encoder', required=True)

parser.add_argument('--decay_rate', default=0.005, type=float, help='the exponential decay rate in the dynamic scheme of gamma')
parser.add_argument('--thres', default=0.015, type=float, help='the threshold (i.e., gamma-0.5) to switch gamma to 0.5')

parser.add_argument('--optimizer', default='Adam', type=str, help='the optimizer to train the model')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='the learning rate to training the model')

parser.add_argument('--n_epochs', default=50, type=int, help='the total number of training epochs with VAMP-2 and dispersion loss optimization')
parser.add_argument('--save_model_interval', default=None, type=int, help='save the model every save_epoch')

parser.add_argument('--train_split', default=0.9, type=float, help='the ratio of training dataset size to full dataset size')
parser.add_argument('--train_batch_size', default=10000, type=int, help='the batch size in training dataloader')
parser.add_argument('--val_batch_size', default=None, type=int, help='the batch size in validation dataloader')

parser.add_argument('--data_directory', type=str, help='the directory storing numpy files of trajectories', required=True)
parser.add_argument('--saving_directory', default='.', type=str, help='the saving directory of training results')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

date_time = datetime.now().strftime("%m_%d_%H_%M")

args.name = (f"{date_time}_memnets_lr_{args.learning_rate}_bsz_{args.train_batch_size}_"
        f"lagtime_{args.lagtimes}_decay_rate_{args.decay_rate}_thres_{args.thres}_"
        f"n_epochs_{args.n_epochs}")

args.log_directory = args.saving_directory+"/{name}/logs".format(name=args.name)
args.model_directory = args.saving_directory+"/{name}/checkpoints".format(name=args.name)

if not os.path.exists(args.model_directory):
    os.makedirs(args.model_directory)
if not os.path.exists(args.log_directory):
    os.makedirs(args.log_directory)

with open(os.path.join(args.log_directory, 'train_args.txt'), 'w') as f:
    f.write(pprint.pformat(state))

def main():

    device = torch.device(args.device)

    data = []
    np_name_list = []
    for np_name in glob.glob(args.data_directory+'/*.npy'):
        data.append(np.load(np_name))
        np_name_list.append(np_name.rsplit('/')[-1])

    set_random_seed(args.seed)

    pre = Preprocessing(dtype=np.float32)
    dataset = pre.create_time_series_dataset(lagtimes=args.lagtimes,data=data)

    val = int(len(dataset)*(1-args.train_split))
    train_data, val_data = random_split(dataset, [len(dataset)-val, val])

    loader_train = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    if val == 0:
        loader_val = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=False)
    else:
        if args.val_batch_size is None or args.val_batch_size >= len(val_data):
            loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
        else:
            loader_val = DataLoader(val_data, batch_size=args.val_batch_size, shuffle=False)

    lobe = MEMLayer(args.encoder_sizes)
    lobe = lobe.to(device=device)

    memnets = MEMNet(lobe=lobe, lagtimes=args.lagtimes, learning_rate=args.learning_rate, device=device, decay_rate=args.decay_rate, thres=args.thres, 
                    save_model_interval=args.save_model_interval)
    memnets_model = memnets.fit(loader_train, n_epochs=args.n_epochs, validation_loader=loader_val).fetch_model()

    validation_lm = memnets.validation_lm
    validation_m0_tilde = memnets.validation_m0_tilde
    validation_log_lambda_hat = memnets.validation_log_lambda_hat
    validation_gamma = memnets.validation_gamma
    validation_Y = memnets.validation_Y
    validation_Y_predict = memnets.validation_Y_predict
    validation_rmse = memnets.validation_rmse

    training_loss = memnets.training_loss
    training_lm = memnets.training_lm
    training_m0_tilde = memnets.training_m0_tilde
    training_log_lambda_hat = memnets.training_log_lambda_hat
    training_gamma = memnets.training_gamma
    training_rmse = memnets.validation_rmse

    np.save((args.model_directory+'/validation_lm.npy'),validation_lm)
    np.save((args.model_directory+'/validation_m0_tilde.npy'),validation_m0_tilde)
    np.save((args.model_directory+'/validation_log_lambda_hat.npy'),validation_log_lambda_hat)
    np.save((args.model_directory+'/validation_gamma.npy'),validation_gamma)
    np.save((args.model_directory+'/validation_Y.npy'),validation_Y)
    np.save((args.model_directory+'/validation_Y_predict.npy'),validation_Y_predict)
    np.save((args.model_directory+'/validation_rmse.npy'),validation_rmse)

    np.save((args.model_directory+'/training_loss.npy'),training_loss)
    np.save((args.model_directory+'/training_lm.npy'),training_lm)
    np.save((args.model_directory+'/training_m0_tilde.npy'),training_m0_tilde)
    np.save((args.model_directory+'/training_log_lambda_hat.npy'),training_log_lambda_hat)
    np.save((args.model_directory+'/training_gamma.npy'),training_gamma)
    np.save((args.model_directory+'/training_rmse.npy'),training_rmse)

    if args.save_model_interval is None:
        torch.save(memnets_model.lobe.state_dict(), args.model_directory+'/model_{}epochs.pytorch'.format(args.n_epochs))

        cvs = memnets_model.transform(data=data,return_cv=True,tau=args.lagtimes[-1])

        memnets_estimator = MEMEstimator(memnets_model,lagtimes=args.lagtimes)
        time_scales = memnets_estimator.fit(data).time_scales
        log_lambda_hat = memnets_estimator.fit(data).log_lambda_hat
        m0_tilde = memnets_estimator.fit(data).m0_tilde
        lm = memnets_estimator.fit(data).lm
        Y = memnets_estimator.fit(data).Y
        Y_predict = memnets_estimator.fit(data).Y_predict
        rmse = memnets_estimator.fit(data).rmse

        dir1 = args.model_directory+'/model_{}epochs_cvs'.format(args.n_epochs)
        dir2 = args.model_directory+'/model_{}epochs_results'.format(args.n_epochs)

        if not os.path.exists(dir1):
            os.makedirs(dir1)
        if not os.path.exists(dir2):
            os.makedirs(dir2)

        np.save((dir2+'/model_{}epochs_time_scales.npy'.format(args.n_epochs)),time_scales)
        np.save((dir2+'/model_{}epochs_log_lambda_hat.npy'.format(args.n_epochs)),log_lambda_hat)
        np.save((dir2+'/model_{}epochs_m0_tilde.npy'.format(args.n_epochs)),m0_tilde)
        np.save((dir2+'/model_{}epochs_lm.npy'.format(args.n_epochs)),lm)
        np.save((dir2+'/model_{}epochs_Y.npy'.format(args.n_epochs)),Y)
        np.save((dir2+'/model_{}epochs_Y_predict.npy'.format(args.n_epochs)),Y_predict)
        np.save((dir2+'/model_{}epochs_rmse.npy'.format(args.n_epochs)),rmse)

        if len(np_name_list) == 1: 
            np.save((dir1+'/cvs_'+np_name_list[0]),cvs)
        else:
            for k in range(len(np_name_list)): 
                np.save((dir1+'/cvs_'+np_name_list[k]),cvs[k])

    else:
        for i in range(len(memnets._save_models)):
            torch.save(memnets._save_models[i].lobe.state_dict(), args.model_directory+'/model_{}epochs.pytorch'.format((i+1)*args.save_model_interval))

            cvs = memnets._save_models[i].transform(data=data,return_cv=True,tau=args.lagtimes[-1])

            memnets_estimator = MEMEstimator(memnets._save_models[i],lagtimes=args.lagtimes)
            time_scales = memnets_estimator.fit(data).time_scales
            log_lambda_hat = memnets_estimator.fit(data).log_lambda_hat
            m0_tilde = memnets_estimator.fit(data).m0_tilde
            lm = memnets_estimator.fit(data).lm
            Y = memnets_estimator.fit(data).Y
            Y_predict = memnets_estimator.fit(data).Y_predict
            rmse = memnets_estimator.fit(data).rmse

            dir1 = args.model_directory+'/model_{}epochs_cvs'.format((i+1)*args.save_model_interval)
            dir2 = args.model_directory+'/model_{}epochs_results'.format((i+1)*args.save_model_interval)

            if not os.path.exists(dir1):
                os.makedirs(dir1)
            if not os.path.exists(dir2):
                os.makedirs(dir2)

            np.save((dir2+'/model_{}epochs_time_scales.npy'.format((i+1)*args.save_model_interval)),time_scales)
            np.save((dir2+'/model_{}epochs_log_lambda_hat.npy'.format((i+1)*args.save_model_interval)),log_lambda_hat)
            np.save((dir2+'/model_{}epochs_m0_tilde.npy'.format((i+1)*args.save_model_interval)),m0_tilde)
            np.save((dir2+'/model_{}epochs_lm.npy'.format((i+1)*args.save_model_interval)),lm)
            np.save((dir2+'/model_{}epochs_Y.npy'.format((i+1)*args.save_model_interval)),Y)
            np.save((dir2+'/model_{}epochs_Y_predict.npy'.format((i+1)*args.save_model_interval)),Y_predict)
            np.save((dir2+'/model_{}epochs_rmse.npy'.format((i+1)*args.save_model_interval)),rmse)

            if len(np_name_list) == 1: 
                np.save((dir1+'/cvs_'+np_name_list[0]),cvs)
            else:
                for k in range(len(np_name_list)): 
                    np.save((dir1+'/cvs_'+np_name_list[k]),cvs[k])

if __name__ == '__main__':
    main()
