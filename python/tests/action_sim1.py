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
import placeholders.Simulation_mock 
from python.placeholder.parameter import Parameter

def action(args):
    # direction
    # 1 --> nest to Tvb
    # 2 --> tvb to nest
    p = Parameter()
    direction = int(args[1])
    
    # 1) Simulation init
    nest = Simulation_mock.NestMock(p.get_nest_path())
    
    # NOTE: Meanwhile...the InterscaleHub is initialized
    
    # 2) Simulation connect
    nest.get_connection_details()
    nest.connect_to_hub()
    
    # 3) Start signal --> simulate or receive, depending on the direction
    if direction == 1:
        nest.simulate()
    elif direction == 2:
        nest.receive()
    
    # 4) Stop signal --> disconnect from hub
    nest.disconnect_from_hub()
    
if __name__ == '__main__':
    sys.exit(action(sys.argv))
