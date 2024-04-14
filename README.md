# BERT Grader

### Environment
```
# Miniconda 3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# conda environment
conda env create -f conf/conda_requirement.yml
pip install -r conf/pip_requirements.txt
# conda create -n bert_grader python==3.8.13
conda activate bert_grader

```

## Dataset and experiments
```
data-speaking/$CORPUS/$DATA_TAG
exp-speaking/$CORPUS/$DATA_TAG/$EXP_TAG
```

## A running example
```
cd src
# Configure the conda environment 
Modify the conda startup method in "path.sh" to your own path

./run_speaking_icnale.sh
```
