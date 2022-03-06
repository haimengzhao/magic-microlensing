#!/bin/bash
#PBS -N train_cde_mdn
#PBS -lselect=1:ncpus=1:mem=64gb:ngpus=1
#PBS -j oe

cd $PBS_O_WORKDIR

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ode-rnn

python train_cde_mdn.py --load 00 --resume 1