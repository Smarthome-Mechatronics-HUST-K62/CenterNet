import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tqdm import tqdm
from loss.loss import loss

def run(train_loader,model,optimizer,device="cpu"):
    print("Training...")
    loader = tqdm(train_loader)
    train_losses = []
    for targets in loader:
        inputs = targets["input"].to(device)
        preds = model(inputs)
        train_loss = loss(preds,targets,device).mean()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        train_losses.append(train_loss.item())
        loader.set_postfix(train_batch_loss=train_losses[-1])
    
    return sum(train_losses)/len(train_losses)