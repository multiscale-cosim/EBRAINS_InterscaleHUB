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
from Interscale_hub.interscale_hub import InterscaleHub
from placeholders.parameter import Parameter

def run_wrapper(args):
    # direction
    # 1 --> nest to Tvb
    # 2 --> tvb to nest
    param = Parameter()
    direction = int(args[1])
    
    #TODO: startet as subprocess by AppCompanion
    # receive steering commands init,start,stop
    
    # 1) init InterscaleHUB
    # includes param setup, buffer creation
    hub = InterscaleHub(param, direction)
    
    # 2) Start signal
    # receive, pivot, transform, send
    hub.start()
    
    # 3) Stop signal
    # disconnect and close ports
    hub.stop()
    stop = True
    
if __name__ == '__main__':
    # args 1 = direction
    sys.exit(run_wrapper(sys.argv))
