#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64
#export PATH=$PATH:/usr/local/cuda-11.3/bin
#export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-11.3

export PYTHONNOUSERSITE=1
export WANDB_DISABLED=true
export WANDB_MODE=offline

eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
conda activate cefr-sp
