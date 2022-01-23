#!/bin/bash
#PBS -N train_scaler
#PBS -lselect=1:ncpus=1:mem=32gb:ngpus=1
#PBS -j oe

cd $PBS_O_WORKDIR

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ode-rnn

python train_scaler.py
