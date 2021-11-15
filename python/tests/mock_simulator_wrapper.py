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
import os
import sys
import placeholders.Simulation_mock as mock 
from python.Application_Companion.common_enums import INTEGRATED_SIMULATOR_APPLICATION as SIMULATOR
from common_enums import DIRECTION


class MockWrapper:
    '''Wrapper/Adapter for (NEST/TVB) simulation mocks.'''
    def __init__(self, args, simulator):
        '''
        initializes with mock simulator and set up their communication
        directions.
        '''
        # get data exchange direction
        # 1 --> nest to Tvb
        # 2 --> tvb to nest
        self.__direction = int(args)  # NOTE: will be changed
        # get mock simulator
        self.__simulator = simulator  # e.g. NEST or TVB
        # initialize the simulator
        self.__execute_init_command()

    def __execute_init_command(self):
        '''
        Executes INIT steering command. Determines local minimum stepsize
        and sends it to the Application Manager.

        NOTE INIT is a system action and thus is done implicitly.
        '''
        # NOTE: Meanwhile...the InterscaleHub starts execution
        # Simulation connect
        self.__simulator.get_connection_details()
        self.__simulator.connect_to_hub()

        # send local minimum step size to Application Manager as a response to
        # INIT
        # NOTE Application Manager expects a string in the following format:
        # {'PID': <int>, 'LOCAL_MINIMUM_STEP_SIZE': <float>}
        pid_and_local_minimum_step_size = \
            {SIMULATOR.PID.name: os.getpid(),
            SIMULATOR.LOCAL_MINIMUM_STEP_SIZE.name: self.__simulator.get_min_delay()}

        # Application Manager will read the stdout stream via PIPE
        # NOTE the communication with Application Manager via PIPES will be
        # changed to some other mechanism
        print(f'{pid_and_local_minimum_step_size}')

    def execute_start_command(self):
        '''
        Executes START steering command.
        Depending on the direction, it simulates or receives.
        '''

        # create a dictionary of choices for the communication direction and
        # their corresponding executions based on the simulator.
        execution_choices = None

        # Case: NEST simulator
        if isinstance(self.__simulator, mock.NestMock):
            execution_choices = {
                        DIRECTION.NEST_TO_TVB: self.__simulator.simulate,
                        DIRECTION.TVB_TO_NEST: self.__simulator.receive}

        # Case: TVB simulator
        if isinstance(self.__simulator, mock.TvbMock):
            execution_choices = {
                        DIRECTION.NEST_TO_TVB: self.__simulator.receive,
                        DIRECTION.TVB_TO_NEST: self.__simulator.simulate}

        # start execution
        try:
            execution_choices[self.__direction]()  # TODO check return codes
        except TypeError:
            # execution_choices is NoneType and could not be initialized
            # because the simulator is not an instance of TVB or NEST
            # TODO log the exception with traceback and terminate with error
            print(f'{self.__simulator} is not an instance of the simulator',
                  file=sys.stderr)
            raise TypeError

        # NOTE InterscaleHub terminates execution with the simulation ends.
        # Thus, implicitly executes the END command.
        # 3) Stop signal --> disconnect from hub
        self.__simulator.disconnect_from_hub()
