#!/bin/bash

# temporary directories and files
rm  -rd test_nest_tvb
mkdir test_nest_tvb
cp ./init_rates.npy  ./test_nest_tvb/init_rates.npy

export PYTHONPATH=$PYTHONPATH:${PWD}/../

mpirun -n 2 python3 app_interscalehub.py 2 &
mpirun -n 1 python3 app_sim1.py 2 &
mpirun -n 1 python3 app_sim2.py 2 &

wait
rm  -rd test_nest_tvb
