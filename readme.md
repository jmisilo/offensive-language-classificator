# Offensive Language Classificator API

The project aims to create a simple API that will allow communication with a fine-tuned model and classifications of the type of tweets/posts - OFFENSIVE or NOT OFFENSIVE. While working on the project, an experiment tracking tool - [W&B](https://wandb.ai/) was used to track performance of the model. Model was trained using a GPU from [Google Colab](https://colab.research.google.com/).

## Results

On the validation set, the model achieved an accuracy of 81.57%, precision of 68.92
% and recall of 74.27%.

## How to run project?

Clone the repository with the command

```
git clone https://github.com/jmisilo/offensive-language-classificator
```

Then go to the directory and install depedencies:

```
cd offensive-language-classificator
pip install -r requirements.txt
```

To run the following command:

```
uvicorn src.app:app --port <PORT>
```

## API documentation

[Documentation]('docs/api_docs.md')

## Data

[Download Data](https://sites.google.com/site/offensevalsharedtask/olid)

[Paper Reference:](https://aclanthology.org/N19-1144.pdf)

Predicting the Type and Target of Offensive Posts in Social Media - Zampieri, Marcos and Malmasi, Shervin and Nakov, Preslav and Rosenthal, Sara and Farra, Noura and Kumar, Ritesh

## Model

[Hugging Face Model Page](https://huggingface.co/siebert/sentiment-roberta-large-english)

[Paper Reference:](https://www.semanticscholar.org/paper/More-than-a-Feeling%3A-Benchmarks-for-Sentiment-Heitmann-Siebert/bfe8c0617ca61496e224380f896c0990fdbf542d)

More than a feeling: Accuracy and Application of Sentiment Analysis - Hartmann, Jochen and Heitmann, Mark and Siebert, Christian and Schamp, Christina