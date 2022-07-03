FROM python:3.8.8

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY /models /app/models
COPY /src /app/src

CMD ["uvicorn", "src.app:app", "--reload", "--port", "5000"]