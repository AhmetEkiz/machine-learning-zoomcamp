# Homework 5 Solutions

I am at 2022 cohort but in self-paced mode!

[**Homework Folder**](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/cohorts/2022/05-deployment)

**[Homework Readme and Solutions Links](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/cohorts/2022/05-deployment/homework.md)**

## Q1

- What's the version of pipenv you installed?

```bash
pipenv --version
```

pipenv, version: 2022.12.19

## Q2

- Use Pipenv to install Scikit-Learn version 1.0.2
- What's the first hash for scikit-learn you get in Pipfile.lock?

The first hash for scikit-learn you get in Pipfile.lock:

"sha256:08ef968f6b72033c16c479c966bf37ccd49b06ea91b765e1cc27afefe723920b”

## Q3

Let's use these models!

- Write a script for loading these models with pickle
- Score this client:

```python
# load model
with open(model_name, 'rb') as f_in: 
    model = pickle.load(f_in)
model

# load  DictVectorizer
with open(dv_name, 'rb') as f_in: 
    dv = pickle.load(f_in)
dv

# client
client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

x = dv.transform([client])

model.predict_proba(x)
y_pred = model.predict_proba(x)[0, 1]
```

Answer is: `0.16213414434326598`

## Q4

Now let's serve this model as a web service

- Install Flask and gunicorn (or waitress, if you're on Windows)
- Write Flask code for serving the model
- Now score this client using `requests`:

**Score:** 0.9282218018527452

### Q4 - Build Docker

New docker build

```docker
FROM python:3.9.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system

COPY ["*.py", "model1.bin", "dv.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict_flask:app"]
```

```bash
# to build container
# zoomcamp is tag, . is current dir. 
docker build -t credit_card .

# to run
docker run -it --rm -p 9696:9696 zoomcamp
```

## Q5

Download the base image `svizor/zoomcamp-model:3.9.12-slim`. You can easily make it by using [docker pull](https://docs.docker.com/engine/reference/commandline/pull/) command.

So what's the size of this base image? : 124.69 MB -125MB

### Creating Dockerfile based on the image prepared from others

```docker
FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system

COPY ["*.py", "model1.bin", "dv.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict_flask:app"]
```

```docker
# to build container
# zoomcamp is tag, . is current dir. 
docker build -t credit_card .

# to run
docker run -it --rm -p 9696:9696 credit_card 
```

## **Q6**

**Score:** 0.9282218018527452
