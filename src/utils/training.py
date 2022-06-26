import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch(model, loader, optimizer, scaler, scheduler, config, device):
    losses = []
    loop = tqdm(loader, total=len(loader))
    
    model.train()

    for inputs, masks, targets in loop:
        inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(inputs, attention_mask=masks, labels=targets)
            
            loss = outputs.loss
            losses.append(loss)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()
            scheduler.step()
            loop.set_description(f'Loss: {loss.item():.4f}')
            loop.refresh()

    return torch.stack(losses).mean().item()

def valid_epoch(model, loader, device):
    losses = []

    model.eval()

    with torch.no_grad():
        for inputs, masks, targets in loader:
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs, attention_mask=masks, labels=targets)
                
                loss = outputs.loss
                losses.append(loss)

    return torch.stack(losses).mean().item()