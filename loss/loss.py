import torch.nn as nn
from loss.loss_components import *


def loss(preds,targets,device):
    """[calculate total loss]
    Args:
        preds ([list of 4-D tensor]): [output of model]
        targets ([dict of tensor]): [ground truth]
    """
    #get pred
    pred_hm, pred_wh, pred_reg, pred_pose_hm, pred_pose_offset, pred_pose_kps = preds
    
    #get ground-truth
    target_hm = targets["hm"].to(device) # (batchsize,1,output_res,output_res) 
    target_reg = targets["reg"].to(device) # (batchsize,max_objs,2)
    target_wh = targets["wh"].to(device) # (batchsize,max_objs,2)
    target_reg_mask = targets["reg_mask"].to(device) #(batchsize,max_objs)
    target_inds = targets["inds"].to(device) #(batchsize,max_objs)
    target_kps = targets["kps"].to(device) #(batchsize,max_objs,num_joints * 2)
    target_kps_mask = targets["kps_mask"].to(device) #(batchsize,max_objs,num_joints * 2)
    target_pose_hm = targets["hm_hp"].to(device) #(batchsize,num_joins,output_res,output_res)
    target_hp_offset = targets["hp_offset"].to(device) #(batchsize,max_objs * num_joints, 2)
    target_hp_mask = targets["hp_mask"].to(device) # (batchsize,max_objs * num_joints)
    target_hp_inds = targets["hp_inds"].to(device) # (batchsize,max_objs * num_joints)
    
    #calculcate heatmap loss (for detection) and pose heatmap loss (for pose estimation)
    pred_hm = torch.clamp(pred_hm.sigmoid_(),min=1e-4,max=1-1e-4)
    pred_pose_hm = torch.clamp(pred_pose_hm.sigmoid_(),min=1e-4,max=1-1e-4)
    hm_loss = heatmap_loss(pred_hm,target_hm)
    pose_hm_loss = heatmap_loss(pred_pose_hm,target_pose_hm)
    
    #calculate wh_loss and reg_loss (offset loss) (for detection) and pose_keypoint_offset loss (for pose estimation)
    wh_loss = offset_loss(pred_wh,target_reg_mask,target_inds,target_wh)
    reg_loss = offset_loss(pred_reg,target_reg_mask,target_inds,target_reg)
    pose_reg_loss = offset_loss(pred_pose_offset,target_hp_mask,target_hp_inds,target_hp_offset)
    
    #calculate 
    keypoint_loss = kp_loss(pred_pose_kps,target_kps_mask,target_inds,target_kps)
    
    loss = hm_loss + 0.1 * wh_loss + reg_loss + pose_hm_loss + pose_reg_loss + keypoint_loss
    
    return loss