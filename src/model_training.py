import torch
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from utils.dataset import get_dataset
from utils.pipeline import get_loader
from utils.config import Config
from utils.training import train_epoch, valid_epoch

if __name__ == '__main__':
    config = Config()

    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    config.num_workers = config.num_workers if is_cuda else 0
    config.pin_memory = config.pin_memory if is_cuda else False

    train_set, valid_set, test_set = get_dataset('data/olid-training-v1.0.tsv')

    train_loader = get_loader(train_set, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=config.pin_memory, shuffle=True)
    valid_loader = get_loader(valid_set, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=config.pin_memory)

    model = AutoModelForSequenceClassification.from_pretrained('siebert/sentiment-roberta-large-english').to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.warmump_steps, 
        num_training_steps=config.epochs * len(train_loader)
    )

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, scheduler, config, device)
        valid_loss = valid_epoch(model, valid_loader, device)

        print(f'Epoch {epoch + 1}: train loss - {train_loss:.4f}, validation loss - {valid_loss:.4f}')