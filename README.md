# BERT Grader

### Environment
```
# Miniconda 3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# conda environment  
conda create -n grader python==3.8.13
conda activate grader
conda env create -f conf/conda_requirement.yml
pip install -r conf/pip_requirements.txt
```

## Dataset and experiments
```
data-speaking/$CORPUS/$DATA_TAG
exp-speaking/$DATA_TAG/$EXP_TAG
```

## A running example
```
cd src
# Configure the conda environment 
Modify the conda startup method in "path.sh" to your own path

./run_speaking_icnale.sh
```
