import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def root():
    return {
        'host': 'OLC API',
        'message': 'Hello! Welcome in Offensive Language Classificator API.'
    }

if __name__ == '__main__':
    # from root
    # uvicorn src.app:app --reload --port 5000
    uvicorn.run('app:app', host='127.0.0.1', port=5000, log_level='info')