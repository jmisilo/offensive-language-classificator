import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")

def text_pipeline(texts):
    """
        texts: list of strings

        returns tokenized_text, attention_mask
    """
    tokenized_text = tokenizer(texts, padding='longest', return_tensors='pt')

    return tokenized_text['input_ids'], tokenized_text['attention_mask']

def collate_fn(batch):
    texts, targets = [], []

    for text, target in batch:
        texts.append(text)
        targets.append(target)

    inputs, masks = text_pipeline(texts)
    
    targets = torch.Tensor(targets).type(torch.long)
    targets = F.one_hot(targets, num_classes=2).type(torch.float32)

    return inputs, masks, targets

def get_loader(dataset, batch_size, shuffle=False, num_workers=0, pin_memory=False):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        shuffle=shuffle,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )