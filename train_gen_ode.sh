#!/bin/bash
#PBS -N train_gen_ode
#PBS -lselect=1:ncpus=20:mem=16gb:ngpus=1
#PBS -j oe

cd $PBS_O_WORKDIR

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ode-rnn

python train_gen_ode.py --load 13658
