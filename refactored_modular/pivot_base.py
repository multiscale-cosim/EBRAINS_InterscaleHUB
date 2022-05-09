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


class PivotBaseClass(metaclass=abc.ABCMeta):
    '''
    Abstract base class for pivot operations. It provides the following
    functionality:
    1) Receives data from an application (e.g. simulator, Insite, etc.).
    2) Transforms the data to the format as required by the receiving
    application.
    3) Sends the data to the destination.
    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return ((hasattr(subclass, 'start') and
                callable(subclass.start) and
                (hasattr(subclass, 'stop') and
                callable(subclass.stop))) or
                NotImplemented)

    @abc.abstractmethod
    def start(self, intracomm):
        """Starts the pivot operations.
        
        Parameters
        ----------
        intracomm : Intra communicator

        Returns
        ------
            return code as int to indicate an un/successful termination.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self):
        """Stops the pivot operations.

        Returns
        ------
            return code as int to indicate an un/successful termination.
        """
        raise NotImplementedError