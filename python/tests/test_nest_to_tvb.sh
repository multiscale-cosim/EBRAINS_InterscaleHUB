#!/bin/bash


rm  -rd test_nest_to_tvb

mkdir ./test_nest_to_tvb 
cp ./init_spikes.npy  ./test_nest_to_tvb/init_spikes.npy

mpirun -n 2 python3 action_hub.py 1
mpirun -n 1 python3 action_sim1.py 1
mpirun -n 1 python3 action_sim2.py 1

wait
rm  -rd test_nest_to_tvb
