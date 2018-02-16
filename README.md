# Machine Learning

A repository made for ICS 435 - Machine Learning.

## Getting Started

Below can be found instructions on how to configure your environment to execute code in the Jupyter notebooks and the code in the ml package.

### Prerequisites

You will need pipenv installed.

If you have pip3 installed, you can use
```bash
pip3 install pipenv
```
or
```bash
pip install pipenv
```

### Installing

Installing all the packages in a virtual environment with pipenv is very easy.
Simply type
```bash
pipenv install
```

Next you will need to setup Jupyter Notebook in your environment. To do this, type
```bash
pipenv shell
python -m ipykernel install --user --name=my-virtualenv-name
```
Where my-virtualenv-name can be found to the left of your terminal prompt after entering the pipenv shell. Once this is done, you can now execute code blocks in the notebooks with Jupyter Notebook from within the pipenv shell!

 
