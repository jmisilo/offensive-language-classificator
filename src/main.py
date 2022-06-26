import torch
from utils.dataset import get_dataset
from utils.pipeline import get_loader
from utils.config import Config

config = Config()

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

config.num_workers = config.num_workers if is_cuda else 0
config.pin_memory = config.pin_memory if is_cuda else False

train_set, valid_set, test_set = get_dataset('data/olid-training-v1.0.tsv')

train_loader = get_loader(train_set, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=config.pin_memory)
valid_loader = get_loader(valid_set, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=config.pin_memory)