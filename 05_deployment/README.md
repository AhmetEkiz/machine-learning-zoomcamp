# Deploy a ML Model to the AWS Elastic Beanstalk

**[For More Details: My Deployment Notes](https://github.com/AhmetEkiz/MLZoomcamp2022/blob/main/05_deployment/05_deployment_notes.md)**

**AWS Elastic Beanstalk** deploys web applications and will notice that we have a lot of requests and it will scale up our service so that it will add more containers.

- [Creating an AWS Account - Machine Learning Bookcamp](https://mlbookcamp.com/article/aws)

# To Deploy our sevice to EB

- We need to install command line interface (CLI) for Elastic Beanstalk, it’s called `awsebcli`

- We want to install it only for this project. For that, we can use `pipenv`

- And this is actually a development dependency so this is something we only need to use to deploy things or like when we are developing so **this is not something we need to have inside the container**.

```bash
# to install AWS Elastic Beanstalk CLI
pipenv install awsebcli --dev

# to activate pipenv environment
pipenv shell

# to see Elastic Beanstalk commands.
eb
```

## Create Local Files for Elastic Beanstalk

AWS EB initialize local files from our docker file.

```bash
# p stands for platform which is docker for us
# r stands for region of our AWS service
# churn-serving is the project name that we named now.
eb init -p docker -r us-east-1 churn-serving
```

This command will create `.elasticbeanstalk` file.

## Test Elastic Beanstalk on Local

```bash
# to run eb
eb local run --port 9696

# from another terminal
python predict_test.py
```

## Deploy to the Cloud as a Web Service

```bash
# create environment
eb create churn-serving-env
```

## Testing web service hosted on the cloud

After environment served successfully: Elastic Beanstalk create an **url** to **get requests** and **response**.

Elastic Beanstalk does’t use our port as a `9696`. It uses port `80`which is default port. So we don’t need to specify the port.

`predict_test.py`file content:

```python
import requests

# from cloud url. 
host = "AWS Elastic Beanstalk URL"
url = f'http://{host}/predict'
```

**WARNING!** : This service is open for all world, so anyone who has this address can use this service and make an requests. So we need to control who accessed this service. 

## Terminate EB

```bash
eb terminate churn-serving-env
```
