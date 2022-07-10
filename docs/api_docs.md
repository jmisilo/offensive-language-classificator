# OLC API docs

## Endpoints

- ```/predict``` - *GET* request

Request body:

```
* text - string, required - text to be classified
```

Response - JSON:

```
* text - string - passed text to classify
* prediction - string - result of the classification - 'Offensive' or 'Not Offensive' value
```

- ```/predictMany``` - *GET* request

Request body:

```
* texts - List[string], required - texts to be classified
```

Response - JSON:

```
* texts - List[string] - passed texts to classify
* predictions - List[string] - results of the classification - List of then 'Offensive' or 'Not Offensive' values
```