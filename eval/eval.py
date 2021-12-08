import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tqdm import tqdm
from loss.loss import loss
from utils.decode import multi_pose_decode,multi_pose_post_process

def save_result(output, batch, results):
    reg = output[3] 
    hm_hp = output[4] 
    hp_offset = output[5]
    dets = multi_pose_decode(
          output[0], output[1], output[2], 
          reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=100)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

    dets_out = multi_pose_post_process(
          dets.copy(), batch['meta']['c'].cpu().numpy(),
          batch['meta']['s'].cpu().numpy(),
          output[0].shape[2], output[0].shape[3])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]


def run(dataset,loader,model,device="cpu"):
    print("Evaluating...")
    loader = tqdm(loader)
    losses = []
    results = {}
    for targets in loader:
        inputs = targets["input"].to(device)
        preds = model(inputs)
        loss = loss(preds,targets,device).mean()
        
        losses.append(loss.item())
        loader.set_postfix(val_batch_loss=losses[-1])
        save_result(preds,targets,results)
    
    mAP = dataset.run_eval(results)
    return sum(losses)/len(losses),mAP