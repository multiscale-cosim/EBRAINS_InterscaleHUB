#!/bin/bash


rm  -rd test_nest_tvb

mkdir test_nest_tvb 
cp ./init_spikes.npy  ./test_nest_tvb/init_spikes.npy

export PYTHONPATH=$PYTHONPATH:${PWD}/../

mpirun -n 2 python3 app_interscalehub.py 1 &
mpirun -n 1 python3 app_sim1.py 1 &
mpirun -n 1 python3 app_sim2.py 1 &

wait
rm  -rd test_nest_tvb
