#!/bin/bash
#PBS -N simulate-data
#PBS -lselect=1:ncpus=2:mem=64gb
#PBS -o /home/hmzhao/latent-ode-microlensing/oe/
#PBS -e /home/zerui603/MDN_lc/log/latent-ode-microlensing/oe/

cd $PBS_O_WORKDIR

conda activate ode-rnn

python simulate.py 100000 5
