import torch
import numpy as np

def convert_pred_to_target_format(pred,inds):
    """

    Args:
        pred ([4-D tensor]): [shape : batchsize,2,heatmap_height,heatmap_width]
        inds ([type]): [shape : batchsize,max_objs if calculate for detection, else shape : batchsize,max_objs * num_joints if calculate for pose estimation]
    """
    pred = pred.permute(0,2,3,1).contiguous() # batchsize,heatmap_height, heatmap_width, 2
    pred = pred.view(pred.shape[0],-1,pred.shape[3]) #batchsize,heatmap_height * heatmap_width, 2
    inds = inds.unsqueeze(2).expand(inds.shape[0],inds.shape[1],pred.shape[2]) #batchsize, max_objs or max_objs * num_joints, 2
    pred = pred.gather(1,inds) #batchsize,max_objs or max_objs*num_joints,2
    return pred
    
    
def heatmap_loss(pred,target,alpha=2,beta=4):
    """[function to calculate loss for heatmap and pose heatmap outputs]

    Args:
        pred ([4-D tensor]): [shape : batchsize, channels, heatmap_height, heatmap_width]
        target ([4-D tensor]): [shape : batchsize, channels, heatmap_height, heatmap_width]
    """
    if isinstance(target,np.ndarray):
        target = torch.from_numpy(target).float()
    
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    
    pos_loss = pos_inds * torch.log(pred) * torch.pow(1 - pred, alpha)
    neg_loss = neg_inds * torch.pow(1-target,beta) * torch.pow(pred, alpha) * torch.log(1 - pred)
    
    num_objs = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_objs == 0:
        loss = -neg_loss
    
    else:
        loss = -(pos_loss+neg_loss)/num_objs
    
    return loss


def offset_loss(pred,mask,inds,target):
    """[this loss is for calculation of wh loss, reg loss ,pose_offset loss]

    Args:
        pred ([4-D tensor]): [shape : batchsize,2,heatmap_height,heatmap_width]
        mask ([2-D tensor]): [shape : batchsize,max_objs if calculate for detection, else shape : batchsize,max_objs * num_joints if calculate for pose estimation]
        inds ([2-D tensor]): [shape : similar shape to mask parameter]
        target ([3-D tensor]): [shape : batchsize,max_objs,2 if calculate for detection, else shape : batchsize,max_objs * num_joints,2 if calculate for pose estimation]
    """
    if isinstance(target,np.ndarray):
        target = torch.from_numpy(target).float()
    
    if isinstance(mask,np.ndarray):
        mask = torch.from_numpy(mask).long()
    
    if isinstance(inds,np.ndarray):
        inds = torch.from_numpy(inds).long()
    
    pred = convert_pred_to_target_format(pred,inds)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = torch.nn.functional.l1_loss(pred * mask, target * mask, size_average = False)
    loss = loss / (mask.sum() + 1e-4)
    return loss
    

def kp_loss(pred,mask,inds,target):
    """[this loss is for calculation of keypoint loss , (distance from keypoint to center in heatmap size)]

    Args:
        pred ([4-D tensor]): [shape : batchsize, num_joints * 2, heatmap_height, heatmap_width]
        mask ([3-D tensor]): [keypoint mask with shape : batchsize, max_objs, num_joints * 2]
        inds ([2-D tensor]): [shape: batchsize,max_objs]
        target ([3-D tensor]): [similar shape to mask]
    """
    if isinstance(target,np.ndarray):
        target = torch.from_numpy(target).float()
    
    if isinstance(mask,np.ndarray):
        mask = torch.from_numpy(mask).long()
    
    if isinstance(inds,np.ndarray):
        inds = torch.from_numpy(inds).long()
        
    pred = convert_pred_to_target_format(pred,inds) #convert to shape : batchsize,max_objs,num_joints * 2
    mask = mask.float()
    loss = torch.nn.functional.l1_loss(pred * mask, target * mask, size_average = False)
    loss = loss / (mask.sum() + 1e-4)
    return loss