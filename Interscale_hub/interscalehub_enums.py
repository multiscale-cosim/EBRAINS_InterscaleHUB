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
    NEST_TO_TVB = 1
    TVB_TO_NEST = 2


@enum.unique
class DATA_BUFFER_STATES(enum.IntEnum):
    """ Enum class for the states of data buffer"""
    HEADER = 0
    READY = 2
