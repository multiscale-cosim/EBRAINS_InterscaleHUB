TODO: adapt
init.sh
test_co-sim.sh

TODO: fix / change
creation of the port files and their names!!!
creation of multiple InterscaleHUBs when adding more NEST regions

STEPS done:
1) test_co-sim.sh
(no mpirun/srun bc no cluster) python3 run_co-sim_test.py ./test_co-sim/ 1 1 $CLUSTER
    - path to new folder
    - int virtual procs
    - int mpi ranks
    - false statement for $cluster
    
2) run_co-sim_test.py
    2a) run_test 
        -- add some params
    2b) run_exploration2D 
        -- do some strange stuff with 'tmp' dict (not necessary at the moment)
    2c) run_exploration 
        -- take general params, new params and params in manager file and store them all in a json file
        TODO: Check if complete (apart from hardcoded stuff in the code)
        TODO: maybe collect this more centrally and remove json file stuff 
    2d) run 
        -- read all params from just created json file
        -- init paths, files, params, logger etc
        TODO: check what is needed and so on
        -- run the 4 processes
    2e) more details:
        run_mpi_nest.sh 
            -  BASEDIR=$(dirname "$0")
            $1 -n $2 python3 $BASEDIR/simulation_Zerlaut.py $3 $4
            - argv=[
                '/bin/sh',
                dir_path,
                mpirun,
                str(param_co_simulation['nb_MPI_nest']),
                str(1),
                results_path,
            ]
        
        run_mpi_tvb.sh
            - BASEDIR=$(dirname "$0")
            $1 -n 1 python3 $BASEDIR/simulation_Zerlaut.py $2 $3
            - argv = [
                '/bin/sh',
                dir_path,
                mpirun,
                str(1),
                results_path,
            ]
        run_mpi_nest_to_tvb.sh
            - BASEDIR=$(dirname "$0")
            $1 -n 1 python3 $BASEDIR/nest_to_tvb.py $2 $3 $4
            - argv=[ '/bin/sh',
                    dir_path,
                    mpirun,
                    results_path,
                    "/translation/spike_detector/"+str(id_spike_detector)+".txt",
                    "/translation/send_to_tvb/"+str(id_proxy[index])+".txt",
                    ]
        run_mpi_tvb_to_nest.sh
            - BASEDIR=$(dirname "$0")
            $1 -n 1 python3 $BASEDIR/tvb_to_nest.py $2 $3 $4 $5
            - argv=[ '/bin/sh',
                    dir_path,
                    mpirun,
                    results_path+"/translation/spike_generator/",
                    str(ids_spike_generator[0]),
                    str(len(ids_spike_generator)),
                    "/../receive_from_tvb/"+str(id_proxy[index])+".txt",
                    ]
        
3) TODO add direction '3/4' to InterscaleHUB 
    -- this means both directions with the extended parameter set and added functionality
