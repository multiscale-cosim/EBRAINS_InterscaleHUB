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
import abc


class InterscaleTransformerBaseClass(metaclass=abc.ABCMeta):
    '''
    Abstract base class for transformation of data to change the scales.
    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return ((hasattr(subclass, 'transform') and
                callable(subclass.transform)) or
                NotImplemented)

    @abc.abstractmethod
    def transform(self, *args, **kwargs):
        """Transforms the data from one format to another .
        
        # TODO discuss what parameters are required for an abstract version of transformation?
        # TODO validate if it transforms the data otherwise return ERROR as response
        # NOTE Followings are taken from rate_to_spike and spike_to_rate functions

        Parameters
        ----------
        data : Any
            Data to be transformed

        count: int
            counter of the number of time of the transformation
            (identify the timing of the simulation)

        buffer: int
            buffer contains id of devices, id of neurons and spike times
        size : int
            size of the data to be read from the buffer for transformation

        Returns
        ------
            returns the data transformed into required format
        """
        raise NotImplementedError
