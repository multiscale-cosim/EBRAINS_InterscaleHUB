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

import Simulation_mock
from python.interscale-hub.interscale_hub import InterscaleHub

class AppCompanionDummy:
    
    def __init__(self, param):
        '''
        Init AC dummy with hardcoded params.
        '''
        self.__param = param
    
    def init_sim(self, path, name):
        '''
        Init the two simulators.
        NOTE: ugly, no error handling and hardcoded for nest and tvb
        NOTE: just a quick hack for the MVP
        '''
        if name == "Nest":
            self.__nest = Simulation_mock.NestMock(path)
        elif name == "Tvb":
            self.__tvb = Simulation_mock.TvbMock(path)
            
    def init_hub(self):
        self.__hub = InterscaleHub(self.__param)
    
    def make_connections(self):
        
    
    def start(self):
        #TODO start all
        pass
    
    def stop_hub(self):
        #TODO: stop all
        pass
    
    
