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


class Communicator():
    '''
    
    '''
    def __init__(self, configurations_manager, log_settings, name, databuffer):
        """Init of parameters
        
        Parameters
        ----------
        intracomm : Intra communicator

        Returns
        ------
            return code as int to indicate an un/successful termination.
        """
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
                                        name=name,
                                        log_configurations=self._log_settings,
                                        target_directory=DefaultDirectories.SIMULATION_RESULTS)
        self.__databuffer = databuffer

    def start(self):
        """Starts the pivot operations.
        
        Parameters
        ----------
        intracomm : Intra communicator

        Returns
        ------
            return code as int to indicate an un/successful termination.
        """
        raise NotImplementedError

    def stop(self):
        """Stops the pivot operations.

        Returns
        ------
            return code as int to indicate an un/successful termination.
        """
        raise NotImplementedError
