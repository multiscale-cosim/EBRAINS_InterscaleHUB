# ------------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor
#  license agreements; and to You under the Apache License, Version 2.0. "
#
# Forschungszentrum Jülich
#  Institute: Institute for Advanced Simulation (IAS)
#    Section: Jülich Supercomputing Centre (JSC)
#   Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
#       Team: Multi-scale Simulation and Design
#
# ------------------------------------------------------------------------------

class Parameter:
    '''
    Parameter class.
    Hardcoded, without any error and safety handling.
    NOTE: 
    List of hardcoded parameter in the InterscaleHub (to be completed)
    - max_events set in InterscaleHub.py
    - min_delay set in Simulation_mock.py
    - simulation params (ids, size) set in Simulation_mock.py
    - tvb to nest params (size_list, list_id) set in pivot.py

    '''
    def __init__(self):
        '''
        init all param for both directions.
        '''
        self.__parameter = {
        "param_nest_to_tvb" : {
        "init": "./tests/init_spikes.npy", 
        "resolution": 0.1, 
        "synch": 100.0, 
        "width": 20.0, 
        "nb_neurons":20}, 
        "param_tvb_to_nest" : {
            "init": "./tests/init_rates.npy", 
            "id_first_spike_detector": 0,
            "nb_spike_generator": 10, 
            "percentage_shared": 0.5, 
            "seed": 42, 
            "nb_synapses":10,
            "function_select":2}
        }
        # path to files containing the MPI port info
        self.__nest_file = '../tests/test_nest_tvb/nest.txt' 
        self.__tvb_file = '../tests/test_nest_tvb/tvb.txt'
        
    def get_nest_path(self):
        return self.__nest_file
    
    def get_tvb_path(self):
        return self.__tvb_file
    
    def get_param(self,direction):
        # direction:
        # 1 --> NEST to TVB
        # 2 --> TVB to NEST
        if direction == 1:
            param = self.__parameter['param_nest_to_tvb']
        elif direction == 2:
            param = self.__parameter['param_tvb_to_nest']
        return param
