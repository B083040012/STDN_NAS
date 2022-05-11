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
        pick_rmse=torch.sqrt(criterion(pick_output[pick_mask], pick_target[pick_mask]))
    else:
        pick_rmse=-1
    # drop part
    if torch.sum(drop_mask)!=0:
        drop_rmse=torch.sqrt(criterion(drop_output[drop_mask], drop_target[drop_mask]))
    else:
        drop_rmse=-1
    return (pick_rmse, torch.sum(pick_mask)), (drop_rmse, torch.sum(drop_mask))