
## Interpretability of CNN

### How to run

Before running check [Virtualenv](https://docs.python-guide.org/dev/virtualenvs/) manual

```
#Create virtualenv (Run in project folder)
virtualenv --python=python3 .env

#Activate environment
source .env/bin/activate

#Install packages (with activated env)
pip install -r requirements.txt
```

[Celeba dataset](https://www.kaggle.com/jessicali9530/celeba-dataset) is used for training and evaluation.

To download dataset use:
```
kaggle datasets download -d jessicali9530/celeba-dataset 
```