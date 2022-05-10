from cmath import inf
import numpy as np
import torch.nn as nn
import torch, logging, time, os, yaml

from utils import AverageMeter
from model import STDN_NAS
from dataloader import STDN_dataloader
from criterion import eval_all, eval_pick_drop


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

        train_portion: ratio of training data/training+validation data
    """
    train_portion=config["training"]["train_portion"]
    logging.info("[Data loading for supernet training...]")
    train_loader, val_loader, null=STDN_dataloader("train",config)
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