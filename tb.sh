#!/bin/bash
#PBS -N tensorboard
#PBS -lselect=1:ncpus=1:mem=4gb:ngpus=0
#PBS -j oe

cd $PBS_O_WORKDIR

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ode-rnn

tensorboard --logdir=/work/hmzhao/tbxdata/
