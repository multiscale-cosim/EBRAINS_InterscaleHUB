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
import placeholders.Simulation_mock as mock
from placeholders.parameter import Parameter
from python.Application_Companion.common_enums import SteeringCommands
from mock_simulator_wrapper import MockWrapper


if __name__ == '__main__':
    '''mock for TVB simulation with steering support.'''
    # initialize parameters
    parameters = Parameter()
    # instantiate TVB mock simulator
    tvb = mock.TvbMock(parameters.get_tvb_path())
    # set direction to the parameter received via Popen subprocess
    direction = sys.argv[1]  # TODO validate the args

    # STEP 1. INIT steering action
    # NOTE INIT is a system action and so is done implicitly when initializes
    # the simulator

    # instantiate the wrapper. and initialize the simulator
    simulator_wrapper = MockWrapper(direction, tvb)

    # STEP 2. START steering command
    # receive steering command from Application Manager via (stdin) PIPE
    user_action_command = input()
    # execute if steering command is START
    if SteeringCommands[user_action_command] == SteeringCommands.START:
        simulator_wrapper.execute_start_command()
        sys.exit(0)
    else:
        # TODO raise and log the exception with traceback and terminate with
        # error if received an unknown steering command
        print(f'unknown steering command: '
              f'{SteeringCommands[user_action_command]}',
              file=sys.stderr)
        sys.exit(1)
