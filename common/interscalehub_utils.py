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
import time

from EBRAINS_RichEndpoint.application_companion.common_enums import Response


def log_exception(logger, log_message, mpi_tag_received):
    try:
        # Raise RunTimeError exception
        raise RuntimeError
    except RuntimeError:
        # Log exception with traceback and mpi tag received
        logger.exception(log_message + str(mpi_tag_received))

def info_log_message(rank, logger, msg):
    "helper function to control the log emissions as per rank"
    if rank == 0:        
        logger.info(msg)
    else:
        logger.debug(msg)

def debug_log_message(rank, logger, msg):
    "helper function to control the log emissions as per rank"
    if rank == 0:        
        logger.debug(msg)
   

def wait_until_buffer_ready(data_buffer_manager, buffer_type, buffer_state):
    while data_buffer_manager.get_at(index=-1,
                                     buffer_type=buffer_type) != buffer_state:
        # buffer is not ready yet
        time.sleep(0.001)
        # continue next iteration
        continue
    
    # buffer is ready
    return Response.OK

