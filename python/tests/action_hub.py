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
from python.Interscale_hub.interscale_hub import InterscaleHub
from python.placeholder.parameter import Parameter

def action(args):
    # direction
    # 1 --> nest to Tvb
    # 2 --> tvb to nest
    param = Parameter()
    direction = int(args[1])
    # param = p.get_param(direction)
    
    # 1) init InterscaleHUB
    # includes param setup, buffer creation
    hub = InterscaleHub(param, direction)
    
    # 2) Open two ports for both simulators...
    # TODO: check with hub __init__(), where to do this?
    #hub.open_ports()
    # ... and accept connections
    #hub.accept_connections()
    
    # 3) Start signal
    # receive, pivot, transform, send
    hub.start()
    
    # 4) Stop signal
    # disconnect and close ports
    hub.stop()
    
if __name__ == '__main__':
    # args 1 = direction
    sys.exit(action(sys.argv))
