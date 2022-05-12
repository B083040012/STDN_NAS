import torch

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
        pick_mse=criterion(pick_output[pick_mask], pick_target[pick_mask])
    else:
        pick_mse=-1
    # drop part
    if torch.sum(drop_mask)!=0:
        drop_mse=criterion(drop_output[drop_mask], drop_target[drop_mask])
    else:
        drop_mse=-1
    return (pick_mse, torch.sum(pick_mask)), (drop_mse, torch.sum(drop_mask))