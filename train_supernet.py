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
import os
import yaml

from utils import AverageMeter
from model import STDN_NAS
from dataloader import STDN_NAS_dataloader

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

def STDN_dataloader(batch_size, num_workers, train_portion, config):
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
    # file_path='data\\slim\\'
    loader=STDN_NAS_dataloader()
    train_nbhd_input, train_flow_input, train_label=loader.sample_stdn( datatype="train", \
                                                                        dataset_size=config["dataset"]["dataset_size"], \
                                                                        att_lstm_num=config["dataset"]["att_lstm_num"], \
                                                                        long_term_lstm_seq_len=config["dataset"]["long_term_lstm_seq_len"],
                                                                        short_term_lstm_seq_len=config["dataset"]["short_term_lstm_seq_len"], \
                                                                        nbhd_size=config["dataset"]["nbhd_size"],
                                                                        cnn_nbhd_size=config["dataset"]["cnn_nbhd_size"])

    # train_nbhd_input=np.load(file_path+'STDN_NAS_train_vol.npy')
    # train_flow_input=np.load(file_path+'STDN_NAS_train_flow.npy')
    # train_label=np.load(file_path+'STDN_NAS_train_label.npy')

    train_dataset=BSSDataConcat(train_nbhd_input, train_flow_input, train_label)
    # test_dataset=BSSDataConcat(test_nbhd_input, test_flow_input, test_label)

    return _get_STDN_dataloader(train_dataset, batch_size, num_workers, train_portion, config["dataset"]["short_term_lstm_seq_len"])

def _get_STDN_dataloader(train_dataset, batch_size, num_workers, train_portion, lstm_seq_num):
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
    
    return train_loader, val_loader

def eval_all(output, target, criterion, threshold):
    mask=target > threshold
    if torch.sum(mask)==0:
        return -1, -1
    loss=criterion(output[mask], target[mask])
    return loss, torch.sum(mask)

def eval_pick_drop(output, target, criterion, threshold):

    # split the pick and drop result and calculate the loss seperately
    pick_output=output[:, 0]
    drop_output=output[:, 1]

    pick_target=target[:, 0]
    drop_target=target[:, 1]
    
    pick_mask=pick_target>threshold
    drop_mask=drop_target>threshold

    # pick part
    if torch.sum(pick_mask)!=0:
        pick_rmse=torch.sqrt(criterion(pick_output[pick_mask], pick_target[pick_mask]))
    else:
        pick_rmse=-1
    # drop part
    if torch.sum(drop_mask)!=0:
        drop_rmse=torch.sqrt(criterion(drop_output[drop_mask], drop_target[drop_mask]))
    else:
        drop_rmse=-1
    return (pick_rmse, torch.sum(pick_mask)), (drop_rmse, torch.sum(drop_mask))


def main():

    with open("parameters.yml", "r") as stream:
        config=yaml.load(stream, Loader=yaml.FullLoader)
    """
    Define initial parameters

        lstm_seq_len: the num of days ranged to predict the output
        threshold: the threshold that ignore the (output,target) loss in criterion
        epoch_max: the num of epoch
        learning rate: the initial lr
        momentum: parameter of the SGD optimizer
        weight decay: parameter of the SGD optimizer
        num_choice: len of the choice list of STDN_NAS
        num_layers: levels of the flow gate mechanism
        device: cpu/gpu
        dataset_size: size of the input data
    """
    lstm_seq_len=config["model"]["lstm_seq_len"]
    threshold=config["dataset"]["threshold"]
    epoch_max=config["training"]["epoch_max"]
    learning_rate=config["training"]["learning_rate"]
    momentum=config["training"]["momentum"]
    weight_decay=config["training"]["weight_decay"]
    weight_decay=float(weight_decay)
    num_choice=config["model"]["num_choice"]
    num_layers=config["model"]["num_layers"]
    device=config["model"]["device"]
    log_dir=config["file"]["log_dir"]
    dataset_size=config["dataset"]["dataset_size"]

    """
    Log File
    """
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt='%m/%d %I:%M:%S %p',
        handlers=[
            logging.FileHandler(log_dir+"STDN_NAS.log"),
            logging.StreamHandler()
        ]
    )

    """
    Get Dataloader

        batch_size: size of each batch
        train_portion: ratio of training data/training+validation data
    """
    batch_size=config["training"]["batch_size"]
    num_workers=config["training"]["num_workers"]
    train_portion=config["training"]["train_portion"]
    logging.info("[Data loading for supernet training...]")
    train_loader, val_loader=STDN_dataloader(batch_size, num_workers, train_portion, config)
    logging.info("Loading Data Complete")

    """
    Define the STDN model
    """
    model=STDN_NAS(lstm_seq_len)
    model=model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_max)
    # print model and training information

    """
    Run Training / Validation Process
    """
    running_start_time=time.time()
    best_val_loss=inf
    logging.info("[Supernet Training Phase...]")
    for epoch in range(epoch_max):
        logging.info("epoch {epoch}/{epoch_max}".format(epoch=epoch, epoch_max=epoch_max))
        total_train_loss=AverageMeter()
        pick_train_loss=AverageMeter()
        drop_train_loss=AverageMeter()
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

            # criterion-total part
            output=model(cnn_tensor_list, flow_tensor_list, nas_choice)
            total_loss, validlen=eval_all(output, target, criterion, threshold)
            if total_loss==-1:
                logging.info("step %d is skip-loss is not valid due to the threshold = %d" %(step, threshold))
                continue
            total_loss.backward()
            optimizer.step()
            total_train_loss.update(total_loss.item(), validlen)

            # criterion-pick and drop part
            (pick_loss, pick_validlen), (drop_loss, drop_validlen)=eval_pick_drop(output, target, criterion, threshold)
            if pick_loss!=-1:
                pick_train_loss.update(pick_loss.item(), pick_validlen)
            if drop_loss!=-1:
                drop_train_loss.update(drop_loss.item(), drop_validlen)

            if step%10==0:
                logging.info("[SuperNet Training]:step {step}/{step_all}".format(step=step+1, step_all=len(train_loader)))
                logging.info("learning rate: %.5f, total_train_loss (mse): %.5f(%.5f)" %(lr, total_loss.item(), total_train_loss.avg))
                logging.info("pick_train_loss_avg (rmse): %.5f, drop_train_loss_avg (rmse): %.5f" %(pick_train_loss.avg, drop_train_loss.avg))

        scheduler.step()

        if train_portion==1:
            continue

        # Supernet Validation
        model.eval()
        total_val_loss=AverageMeter()
        pick_val_loss=AverageMeter()
        drop_val_loss=AverageMeter()

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

                # criterion-total part
                total_loss, validlen=eval_all(output, target, criterion, threshold)
                if total_loss==-1:
                    continue
                total_val_loss.update(total_loss.item(), validlen)

                # criterion-pick and drop part
                (pick_loss, pick_validlen), (drop_loss, drop_validlen)=eval_pick_drop(output, target, criterion, threshold)
                if pick_loss!=-1:
                    pick_val_loss.update(pick_loss.item(), pick_validlen)
                if drop_loss!=-1:
                    drop_val_loss.update(drop_loss.item(), drop_validlen)

            if total_val_loss.avg<best_val_loss:
                best_val_loss=total_val_loss.avg
                logging.info("%%%%%%%%%%%%%%%%%%%%%%%%")
                logging.info("Best checkpoint for val_loss %.5f" %(total_val_loss.avg))
                ckpt_file=os.path.join(log_dir+"checkpoint.pth")
                torch.save(model.state_dict(), ckpt_file)
                logging.info("checkpoint file saved")
                logging.info("%%%%%%%%%%%%%%%%%%%%%%%%")
            logging.info("[SuperNet Validation] epoch: %d/%d, val_loss_avg: %.5f, best_val_loss (mse): %.5f" %(epoch, epoch_max, total_val_loss.avg, best_val_loss))
            logging.info("pick_val_loss_avg (rmse): %.5f, drop_val_loss_avg (rmse): %.5f" %(pick_val_loss.avg, drop_val_loss.avg))
        logging.info("**********************************")
        logging.info("End of epoch {epoch}/{epoch_max}".format(epoch=epoch+1, epoch_max=epoch_max))
        logging.info("===========================================")

    running_end_time=time.time()
    logging.info("total training time: %.5f sec" %(running_end_time-running_start_time))

if __name__=='__main__':
    main()