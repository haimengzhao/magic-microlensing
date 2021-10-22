#!/bin/bash
#PBS -N simulate
#PBS -lselect=1:ncpus=20:mem=2gb
#PBS -j oe

cd $PBS_O_WORKDIR

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ode-rnn

python simulate.py 100000 5 1 20 ./log/log.txt
