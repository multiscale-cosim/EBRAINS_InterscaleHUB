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

import sys
import placeholders.Simulation_mock as mock
from placeholders.parameter import Parameter


def run_wrapper(args):
    # direction
    # 1 --> nest to Tvb
    # 2 --> tvb to nest
    p = Parameter()
    direction = int(args[1])
    # param = p.get_param(direction)
    stop = False
    
    while not (stop):
        
        #TODO: receive steering commands from AppCompanion
        command = 1
        
        if command == 1:
            # 1) Simulation init
            tvb = mock.TvbMock(p.get_tvb_path())
            
            # NOTE: Meanwhile...the InterscaleHub is initialized
            
            # 2) Simulation connect
            tvb.get_connection_details()
            tvb.connect_to_hub()
            
            # 3) simulate or receive, depending on the direction
            if direction == 1:
                tvb.receive()
            elif direction == 2:
                tvb.simulate()
            
            # 4) Stop signal --> disconnect from hub
            tvb.disconnect_from_hub()
            stop = True
        #elif command == 2:
            # placeholder start
            # if direction == 1:
            #     tvb.receive()
            # elif direction == 2:
            #     tvb.simulate()
        #elif command == 3:
            # placeholder stop
            # tvb.disconnect_from_hub()
            # stop = True
    
if __name__ == '__main__':
    # args 1 = direction
    sys.exit(run_wrapper(sys.argv))