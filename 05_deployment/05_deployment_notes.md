# 5. Deploying Machine Learning Models README

This week, we **deploy** our **Churn Service** which is **detect customers** who are likely to churn or stopping to use the company service. So, this service tells the marketing department to make a marketing campaign for customers who will stop to use the company service. In order to **deploy** our **services**, we use some **tools** and services such as **Docker**, **Pipenv**, and **AWS Elastic Beanstalk**. After deploying, our API can answer requests with prediction.



**Overview of Our Churn Service:**

![Overview of Our Churn Service ](images/Untitled.png)

**Deployment Environment Our Service:**

![Deployment Environment Our Service.](images/Untitled%201.png)

# 1. Environment

When we are developing our ML model, we can use **Conda** environment to try packages. And because of our services will work on **Linux** systems, we use **WSL (*Windows Subsystem for Linux*)** and **Ubuntu**. 

> 📌 Basically, **WSL** and **Conda** environment is used for development and we run all of our **bash** commands on **WSL**.

There is a awesome tutorial to setup your **WSL** and **other packages** like **Docker**, **Pipenv**, **VSCode**, **Jupyter Notebook** on **WSL**.

- [MemoonaTahira/MLZoomcamp2022 **Setting_up_WSL+Docker.md**](https://github.com/MemoonaTahira/MLZoomcamp2022/blob/main/Notes/Week_5-flask_and_docker_for_deployment/Setting_up_WSL%2BDocker.md)

- [MLZoomcamp2022/Notes/Week_5-flask_and_docker_for_deployment at main · MemoonaTahira/MLZoomcamp2022](https://github.com/MemoonaTahira/MLZoomcamp2022/tree/main/Notes/Week_5-flask_and_docker_for_deployment)

# 2. Saving and Loading The Model

After developing our ml model, we save model to use another time. To save that we can use `pickle` that is built-in module in Python. Because of our ml model is a `Sklearn` object, we must have to `Sklearn` library installed on our environment.



from `save_and_load_churn_model.ipynb` :

```python
# to save the model and dictionary vectorizer
output_file = f'model_C={C}.bin'
with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, model), f_out)

# to load the model and dictionary vectorizer
with open(input_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)
```

# 3.Web Service, Flask and API

We need a web service to communicate with our other services or who wants to use our services.

> A web service is a method used to communicate between electronic devices.

In order to predict and send the results to the user we use `POST`method which is used to send data to a server to create/update a resource. [Source](https://www.w3schools.com/tags/ref_httpmethods.asp)

from `predict_flask.py` :

```python
@app.route('/predict', methods=['POST']) # send prediction result to customer
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

        # we need to conver numpy data into python data
    result = {
        'churn_probability': float(y_pred), 
        'churn': bool(churn)
    }
    return jsonify(result) # send back the data in json format to the user
```

To run locally and test the code, modify and run `predict_test.py` :

```python
import requests

url = 'http://localhost:9696/predict'

customer_id = 'xyz-123'
customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 12,
    "monthlycharges": 29.85,
    "totalcharges": (12 * 29.85)
}

response = requests.post(url, json=customer).json()
print(response)

if response['churn'] == True:
    print('sending promo email to %s' % customer_id)
else:
    print('not sending promo email to %s' % customer_id)
```

```bash
# to activate our conda environment
conda activate ml-zoomcamp

python predict_flask.py

python predict_test.py
```

# 4. Serving the Flask App with [Gunicorn](https://gunicorn.org/)

Running the **Flask App** will give you a warning **“WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.”** To fix this warning, we will use [gunicorn](https://gunicorn.org/) 

![The warning](images/Untitled%202.png)

> ***Gunicorn** 'Green Unicorn' is a Python **WSGI HTTP Server** for **UNIX**.*

[Gunicorn](https://gunicorn.org/) works on Linux and Mac, not on Windows, so you can use an alternative which is Waitress on Windows. So, when running on Windows you can use waitress to try production BUT we will deploy on a Linux environment, therefore we will use gunicorn.

![Untitled](images/Untitled%203.png)

```bash
# install gunicorn on mac or linux to serve our Flask app
pip install gunicorn

# to run WGSI server 
gunicorn --bind 0.0.0.0:9696 predict_flask:app

# install waitress on windows to serve our Flask app
pip install waitress

waitress-serve --listen=0.0.0.0:9696 predict_flask:app
```

![Untitled](images/Untitled%204.png)


⚠️ When I say serving on Windows, you can serve locally to try if you are working on Windows, not WSL or Linux. Because when we put on production on AWS, we need to serve on a Linux environment. Therefore we use WSL and serve on a Linux environment.

</aside>

To run a WGSI server, we run command `gunicorn --bind 0.0.0.0:9696 predict_flask:app`. So, in this command we need to tell where is our Flask app, in order to do that we write `predict_flask:app` which `predict_flask`is the name of the Flask app Python file that we want to run. ([Source from the video](https://youtu.be/Q7ZWPgPnRz8?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&t=617))

## 4.1 Test Served Flask App

To test your served Flask app, use `predict_test.py.` Before that, you may need to modify the file:

```python
url = 'http://localhost:9696/predict'
```

Run the command:

```bash
python predict_test.py
```

![Untitled](images/Untitled%205.png)

# 5. Python Virtual Environment: Pipenv

We tested our codes **locally**. Now, we need to prepare environments for **deployment** to the **cloud.** 

We use libraries for our projects and these libraries may conflict with each other or some functions don't work due to new releases or because different versions are used from development. To avoid it, we need to use virtual environments. We will use **pipenv.**

> *pipenv is similar conda and poetry. Pipenv is more uses from python community.*

Other options:

- venv
- conda
- pipenv
- poetry

![Untitled](images/Untitled%206.png)

## 5.1 install pipenv

```bash
# to install pipenv
pip install pipenv
```

After installing pipenv, we need to install the libraries that we use in our project:

```python
pipenv install numpy scikit-learn==0.24.2 flask
```

- Note that using the **pipenv** command we made two files named ***`Pipfile`*** and ***`Pipfile.lock`***.

![created files with pipenv install](images/Untitled%207.png)

created files with pipenv install

- **`pipfile`** : which packages we need.
- **[dev-packages]** :  packages that you only need for development so you only want to have them on your laptop but you don’t want to have them when you deploy your service production environment.

**Pipfile**:

![Pipfile](images/Untitled%208.png)

## 5.2 Activate Pipenv Environment

```bash
# to activate pipenv environment
pipenv shell
```

![Untitled](images/Untitled%209.png)

![Untitled](images/Untitled%2010.png)

## 5.3 Run Gunicorn

```bash
# to activate environments
conda activate ml-zoomcamp
pipenv shell

# start web app
gunicorn --bind 0.0.0.0:9696 predict:app

# to run test
python predict_test.p
```

### An error (for an example): Trying to unpickle estimator DictVectorizer from version 1.1.3 when using version 0.24.2. This might lead to breaking code or invalid results.

![Untitled](images/Untitled%2011.png)

This was happened because, in the beginning we used scikit-learn==1.1.3 then we use 0.24.2 version of that library to deploy. 

### An error 2: No module named 'requests'

> File "/mnt/h/My Drive/Projects/ml_zoomcamp_22/05_deployment/predict_test.py", line 1, in <module>
> import requests
> ModuleNotFoundError: No module named 'requests'

![Untitled](images/Untitled%2012.png)

> “Requests is not a built in module (does not come with the default python installation), so you will have to install it:” ([Source](https://stackoverflow.com/a/17309309/8654414))

- test the code:

```bash
python predict_test.py
```

# 6. [Environment management: Docker](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/05-deployment/06-docker.md)

![Untitled](images/Untitled%2013.png)


📌 *To isolate more our project file from our system machine, there is an option named **Docker**. With **Docker** you are able to pack all your project is a system that you want and run it in any system machine. For example if you want **Ubuntu 20.4** you can have it in a mac or windows machine or other operating systems.*

</aside>

- First, we need to build a Docker Image.

We search for Docker Python tags on [https://hub.docker.com/_/python](https://hub.docker.com/_/python) :

![Untitled](images/Untitled%2014.png)

- Then run Python Docker Container:

```bash
docker run -it --rm python:3.8.12-slim
```

![Untitled](images/Untitled%2015.png)

![Untitled 16.png](images/Untitled%2016.png)

![Untitled](images/Untitled%2017.png)

## 6.1 Build and Run an Image for Our Project

From our `Dockerfile`file: (we need to remove comment sections)

```bash
# First install the python 3.8, the slim version have less size
FROM python:3.8.12-slim

# Install pipenv library in Docker 
RUN pip install pipenv

# we have created a directory in Docker named app and we're using it as work directory 
WORKDIR /app                                                                

# Copy the Pip files into our working derectory 
COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependecies we had from the project and deploy them 
RUN pipenv install --deploy --system

# Copy any python files and the model we had to the working directory of Docker 
COPY ["*.py", "churn-model.bin", "./"]

# We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 9696

# If we run the Docker image, we want our churn app to be running
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "churn_serving:app"]
```

```bash
# to build container
# zoomcamp is tag, . is current dir. 
docker build -t zoomcamp .

# to run - but it will not communicate
docker run -it --rm zoomcamp
```

- **Entrypoint** is the default command that is executed when we docker run. So, If we don't put the last line `ENTRYPOINT` , we will be in a python shell.

**The Expose Illustration**:

![The Expose Illustration](images/Untitled%2018.png)

# 7. [Deployment to the cloud: AWS Elastic Beanstalk](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/05-deployment/07-aws-eb.md)

**Deployment** is the most crucial part of the programming. Because if you have a model and you can't run it for your users, you don't have any results. **Results-oriented** work is the key. To make it happen, we will deploy our model on **AWS Elastic Beanstalk** which **deploys web applications**.

**Elastic Beanstalk** adds more docker container that contains our services to answer requests when **a lot of requests** comes in.

> *Elastic Beanstalk is a service for deploying and scaling web applications and services. Upload your code and Elastic Beanstalk automatically handles the deployment—from capacity provisioning, load balancing, and auto scaling to application health monitoring.*

**To create an AWS Account:**

- [Creating an AWS Account - Machine Learning Bookcamp](https://mlbookcamp.com/article/aws)

![Untitled](images/Untitled%2019.png)

- We need to install especially command line interface (CLI) for Elastic Beanstalk, it’s called`awsebcli`
- We want to install it only for this project. For that, we can use `pipenv`
- And this is actually a dev dependency so this is something we only need to use to deploy things or like when we are developing so this is not something we need to have inside the container.

```bash
# to install AWS Elastic Beanstalk CLI
pipenv install awsebcli --dev

# to activate pipenv environment
pipenv shell

# elastic beanstalk
eb
```

![Untitled](images/Untitled%2020.png)

# To Deploy our sevice to EB

```bash
# p stands for platform which is docker for us
# churn-serving our project name that we named now
# r stands for region
eb init -p docker -r us-east-1 churn-serving
```

![Untitled](images/Untitled%2021.png)

![Untitled](images/Untitled%2022.png)

## Test Elastic Beanstalk

```bash
# to run eb locally
eb local run --port 9696

# from another terminal
python predict_test.py
```

![Untitled](images/Untitled%2023.png)

![Untitled](images/Untitled%2024.png)

# Web Service on Cloud

```bash
# create environment
eb create churn-serving-env
```

![Untitled](images/Untitled%2025.png)

### An error: ERROR Instance deployment: Both 'Dockerfile' and 'Dockerrun.aws.json' are missing in your source bundle. Include at least one of them. The deployment failed.

I named `Dockerfile` as a `dockerfile`. So it must be named `Dockerfile`

- [https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/single-container-docker-configuration.html](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/single-container-docker-configuration.html)

> Create a `Dockerfile` to have Elastic Beanstalk build and run a custom image. This file is optional, depending on your deployment requirements. For more information about the `Dockerfile` see [Dockerfile reference](https://docs.docker.com/engine/reference/builder/) on the Docker website.

**This is from website. AWS console:**

![This is from website. aws console ](images/Untitled%2026.png)

# Testing web service hosted on the cloud

After environment served successfully: Elastic Beanstalk create an url to get requests and response. 

Elastic Beanstalk does’t use our port as a `9696`. It uses port `80`which is default port. So we don’t need to specify the port. 

![Untitled](images/Untitled%2027.png)

⚠️WARNING : This service is open for all world, so anyone who has this address can use this service and make an requests. So we need to control who accessed this service. We don’t know how to yet.

![Untitled](images/Untitled%2028.png)

# Terminate EB

```bash
eb terminate churn-serving-env
```