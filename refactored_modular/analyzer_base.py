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


class ScienceAnalyzerBaseClass(metaclass=abc.ABCMeta):
    '''
    Abstract base class for analysis of data.
    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return ((hasattr(subclass, 'analyze') and
                callable(subclass.analyze)) or
                NotImplemented)

    @abc.abstractmethod
    def analyze(self, data, time_start, time_stop, **kwargs):
        """analyzes the data for a given time interval and returns the results.
        
        # TODO Discuss what parameters are required for an abstract version of analysis?
        # TODO Validate if it analyze the data otherwise return ERROR as response
        # NOTE Followings are taken from rate_to_spike and spike_to_rate functions

        Parameters
        ----------
        data : Any
            Data to be analyzed

        time_start: int
           time to start the analysis

        time_stop: int
           time to stop the analysis

        variation : bool
            boolean for variation of rate

        windows: float
            the window to compute rate

        Returns
        ------
            returns the analyzed data
        """
        raise NotImplementedError
