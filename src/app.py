import os
import torch
import uvicorn
from dotenv import dotenv_values
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification

try:
    from src.utils.pipeline import text_pipeline
except:
    from utils.pipeline import text_pipeline

app = FastAPI()

device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSequenceClassification.from_pretrained('siebert/sentiment-roberta-large-english').to(device)

try:
    state_dict = torch.load(os.path.join('..', 'models', 'model.pt'), map_location=device)
except:
    state_dict = torch.load(os.path.join('models', 'model.pt'), map_location=device)

model.load_state_dict(state_dict)
model.eval()

secret_config = dotenv_values()

@app.get('/')
def root():
    return {
        'host': 'OLC API',
        'message': 'Hello! Welcome in Offensive Language Classificator API.'
    }

class Post(BaseModel):
    text: str

@app.get('/predict')
def predict(post: Post) -> str:

    tokenized_text, attention_mask = text_pipeline([post.text])
    prediction = model(input_ids=tokenized_text, attention_mask=attention_mask)

    return {
        'text': post.text, 
        'prediction': 'Offensive' if torch.argmax(prediction.logits).item() else 'Not Offensive'
    }

if __name__ == '__main__':
    # from root
    # uvicorn src.app:app --reload --port 5000
    port = int(secret_config['PORT']) if secret_config['PORT'] else 5000

    uvicorn.run('app:app', host='127.0.0.1', port=port, reload=True)