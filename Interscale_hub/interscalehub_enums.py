# ------------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements; and to You under the Apache License,
# Version 2.0. "
#
# Forschungszentrum Jülich
# Institute: Institute for Advanced Simulation (IAS)
# Section: Jülich Supercomputing Centre (JSC)
# Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
# Team: Multi-scale Simulation and Design
# ------------------------------------------------------------------------------
import enum


@enum.unique
class DATA_EXCHANGE_DIRECTION(enum.IntEnum):
    """ Enum class for communication direction"""
    # TODO refactor all usecases to replace with more general enums
    NEST_TO_TVB = 1
    TVB_TO_NEST = 2
    NEST_TO_LFPY = 3


@enum.unique
class DATA_BUFFER_STATES(enum.IntEnum):
    """ Enum class for the states of data buffer"""
    READY_TO_SEND = 0
    READY_TO_RECEIVE = 1
    READY_TO_TRANSFORM = 2
    WAIT = 3
    TERMINATE = 4
    

@enum.unique
class DATA_BUFFER_TYPES(enum.IntEnum):
    """ Enum class for different buffer types"""
    INPUT = 0
    OUTPUT = 1