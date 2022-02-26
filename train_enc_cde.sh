#!/bin/bash
#PBS -N train_enc_cde_log
#PBS -lselect=1:ncpus=1:mem=64gb:ngpus=1
#PBS -j oe

cd $PBS_O_WORKDIR

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ode-rnn

python train_enc_cde.py
