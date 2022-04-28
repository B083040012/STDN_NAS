import argparse
from cProfile import label
from cmath import inf
from ctypes import util
import numpy as np
import torch.utils.data as data
import random
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import logging
import time

from utils import AverageMeter
from model import STDN_NAS

class BSSDataConcat(data.Dataset):
    def __init__(self, nbhd, flow, label, tranform=None):
        self.transform= tranform
        self.__xs=nbhd
        self.__ys=flow
        self.__zs=label

    def __getitem__(self, index):
        return (self.__xs[index], self.__ys[index], self.__zs[index])
    
    def __len__(self):
        return len(self.__xs)

def STDN_dataloader(batch_size, num_workers, train_portion, lstm_seq_num):
    """
    define the dataloader of STDN input
    1. training & testing data:
        a. nbhd_input
        b. flow_input
    2. training & testing label
    both load by .npz file

    data_size:
        slim dataset:
            vol:  (1000,7,2,7,7)
            flow: (1000,7,4,7,7)
            label:(1000,2)
            7 lstm_seq_len, 1000 input data, 2 or 4 channel, 7*7 image
        complete dataset:
            vol:  (287400,7,2,7,7)
            flow: (287400,7,4,7,7)
            label:(287400,2)
    """
    file_path='data\\slim\\'

    train_nbhd_input=np.load(file_path+'STDN_NAS_train_vol.npy')
    test_nbhd_input=np.load(file_path+'STDN_NAS_test_vol.npy')
    train_flow_input=np.load(file_path+'STDN_NAS_train_flow.npy')
    test_flow_input=np.load(file_path+'STDN_NAS_test_flow.npy')
    train_label=np.load(file_path+'STDN_NAS_train_label.npy')
    test_label=np.load(file_path+'STDN_NAS_test_label.npy')

    train_dataset=BSSDataConcat(train_nbhd_input, train_flow_input, train_label)
    test_dataset=BSSDataConcat(test_nbhd_input, test_flow_input, test_label)

    return _get_STDN_dataloader(train_dataset, test_dataset, batch_size, num_workers, train_portion, lstm_seq_num)

def _get_STDN_dataloader(train_dataset, test_dataset, batch_size, num_workers, train_portion, lstm_seq_num):
    """
    convert STDN dataset into dataloader type, 
    and split the validation data from training data
    """
    if train_portion != 1:
        train_len = len(train_dataset)
        indices = list(range(train_len))
        # shuffle index to enhance the randomness
        random.shuffle(indices)
        split = int(np.floor(train_portion * train_len))
        train_idx, val_idx = indices[:split], indices[split:]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader=DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
            pin_memory=True)
        val_loader=DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=val_sampler,
            pin_memory=True)
    else:
        train_loader=DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True)
        val_loader = None

    test_loader=DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True)
    
    return train_loader, val_loader, test_loader


def main():

    """
    Define initial parameters

        lstm_seq_len: the num of days ranged to predict the output
        epoch_max: the num of epoch
        learning rate: the initial lr
        momentum: parameter of the SGD optimizer
        weight decay: parameter of the SGD optimizer
        num_choice: len of the choice list of STDN_NAS
        num_layers: levels of the flow gate mechanism
        device: cpu/gpu
    """
    lstm_seq_len=7
    epoch_max=5
    learning_rate=0.025
    momentum=0.9
    weight_decay=3e-4
    num_choice=3
    num_layers=3
    device='cuda'

    """
    Log File
    """
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt='%m/%d %I:%M:%S %p',
        handlers=[
            logging.FileHandler("STDN_NAS.log", mode='w'),
            logging.StreamHandler()
        ]
    )

    """
    Get Dataloader

        batch_size: size of each batch
        train_portion: ratio of training data/training+validation data
    """
    batch_size=20
    num_workers=0
    train_portion=0.8
    train_loader, val_loader, test_loader=STDN_dataloader(batch_size, num_workers, train_portion, lstm_seq_len)
    logging.info("Loading Data Complete")

    """
    Define the STDN model
    """
    model=STDN_NAS(lstm_seq_len)
    model=model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_max)

    """
    Run Training / Validation Process
    """
    running_start_time=time.time()
    best_val_loss=inf

    for epoch in range(epoch_max):
        logging.info("===========================================")
        logging.info("epoch {epoch}/{epoch_max}".format(epoch=epoch, epoch_max=epoch_max))
        train_loss=AverageMeter()
        lr=optimizer.param_groups[0]["lr"]
        
        # SuperNet Training
        model.train()
        for step, (nbhd, flow, label) in enumerate(train_loader):
            # resize the tensor to [lstm_seq_num, batch_size, channel, size, size]
            cnn_tensor_list=nbhd.permute(1,0,2,3,4)
            flow_tensor_list=flow.permute(1,0,2,3,4)
            target=label.to(device).float()
        
            optimizer.zero_grad()
            # random sample the NAS choice block
            nas_choice=list(np.random.randint(num_choice, size=num_layers))
            output=model(cnn_tensor_list, flow_tensor_list, nas_choice)
            loss=torch.sqrt(criterion(output, target))
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), nbhd.size(0))
            if step%10==0:
                logging.info("[SuperNet Training]:step {step}/{step_all}".format(step=step+1, step_all=len(train_loader)))
                logging.info("learning rate: %.5f, train_loss: %.5f(%.5f)" %(lr, loss.item(), train_loss.avg))

        scheduler.step()

        if train_portion==1:
            continue

        # Supernet Validation
        model.eval()
        val_loss=AverageMeter()

        logging.info("**********************************")
        logging.info("validation step for epoch {epoch}".format(epoch=epoch))
        with torch.no_grad():
            for step, (nbhd, flow, label) in enumerate(val_loader):
                # resize the tensor to [lstm_seq_num, batch_size, channel, size, size]
                cnn_tensor_list=nbhd.permute(1,0,2,3,4)
                flow_tensor_list=flow.permute(1,0,2,3,4)
                target=label.to(device).float()

                # random sample the NAS choice block
                nas_choice=list(np.random.randint(num_choice, size=num_layers))
                output=model(cnn_tensor_list, flow_tensor_list, nas_choice)
                loss=torch.sqrt(criterion(output, target))
                val_loss.update(loss.item(), nbhd.size(0))
            if val_loss.avg<best_val_loss:
                best_val_loss=val_loss.avg
                # need to save the best checkpoint(?
                logging.info("%%%%%%%%%%%%%%%%%%%%%%%%")
                logging.info("Best checkpoint for val_loss %.5f" %(val_loss.avg))
                logging.info("%%%%%%%%%%%%%%%%%%%%%%%%")
            logging.info("[SuperNet Validation] epoch: %d/%d, val_loss: %.5f, best_val_loss: %.5f" %(epoch, epoch_max, val_loss.avg, best_val_loss))
        logging.info("**********************************")
        logging.info("End of epoch {epoch}/{epoch_max}".format(epoch=epoch+1, epoch_max=epoch_max))
        logging.info("===========================================")

    running_end_time=time.time()
    logging.info("total training time: %.5f sec" %(running_end_time-running_start_time))

if __name__=='__main__':
    main()