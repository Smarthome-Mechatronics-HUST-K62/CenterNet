import os
import sys
import inspect
import argparse
import torch
import yaml
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from torch.utils.data import DataLoader
from model.centernet import CenterNet
from dataset.coco_pose import COCOHP
from eval import eval
import trainer


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset",type=str,required=True,help="Path to train image directory")
    parser.add_argument("--train_annotations",type=str,required=True,help="Path to train keypoint annotation file (.json)")
    parser.add_argument("--val_dataset",type=str,required=True,help="Path to val image directory")
    parser.add_argument("--val_annotations",type=str,required=True,help="Path to val keypoint annotation file (.json)")
    parser.add_argument("--output_dir",type=str,required=True,help="Directory to save checkpoints")
    parser.add_argument("--pretrained",type=str,required=False,default=None,help="Path to checkpoint (.pth),if None -> train from beginning (epoch 0)")
    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    
    train_img_dir = args.train_dataset
    train_annotation_path = args.train_annotations
    val_img_dir = args.val_dataset
    val_annotation_path = args.val_annotations
    checkpoint_path = args.pretrained
    if os.path.isdir(args.output_dir) is not True:
        os.mkdir(args.output_dir)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device : ",device)
    
    with open("train.yaml","r") as cfg:
        train_cfg = yaml.load(cfg, Loader=yaml.FullLoader)

    #Get train and val dataloader
    train_dataset = COCOHP(train_img_dir,train_annotation_path,train_mode=True,model_inp_size=train_cfg["IMG_SIZE"],down_ratio=train_cfg["DOWN_RATIO"])
    train_loader = DataLoader(train_dataset,batch_Size=train_cfg["BATCH_SIZE"],num_workers=train_cfg["NUM_WORKERS"])
    
    val_dataset = COCOHP(val_img_dir,val_annotation_path,train_mode=False,model_inp_size=train_cfg["IMG_SIZE"],down_ratio=train_cfg["DOWN_RATIO"])
    val_loader = DataLoader(val_dataset,batch_size=1,num_workers=1)

    #Get model
    model = CenterNet(train_cfg["DOWN_RATIO"]).to(device)
    if checkpoint_path is not None:
        pass

    else:
        start_epoch = 1
        train_loss_hist = []
        val_loss_hist = []
        best_mAP = 0
        best_epoch = 0
    
    #Get optimizer
    optimizer = torch.optim.Adam(model.parameters(),float(train_cfg["LR"]))
    
    for epoch in range(start_epoch,train_cfg["EPOCHS"]):
        print("Epoch : {}, lr : {}".format(epoch,optimizer.param_groups[0]['lr']))
        train_loss = trainer.run(val_loader,model,optimizer,device)
        val_loss, val_mAP = eval.run(val_dataset,val_loader,model,device)
        print("Train Loss : {}, Val Loss : {}, Val mAP : {}".format(train_loss,val_loss,val_mAP))
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            best_epoch = epoch
            checkpoints = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss_hist": train_loss_hist,
                "val_loss_hist:": val_loss_hist,
                "best_mAP": best_mAP,
                "epoch:": epoch,
                "lr": optimizer.param_groups[0]['lr']
            }
            torch.save(checkpoints,os.path.join(args.output_dir,"best_model_at_epoch_"+str(epoch)+".pth"))
        print("Best mAP : {} at epoch : {}".format(best_mAP,best_epoch))
        
        #lr scheduler:
        if epoch in train_cfg["LR_STEP"]:
            lr = float(train_cfg["LR"]) * (0.1 ** (train_cfg["LR_STEP"].index(epoch)+1))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
    
    