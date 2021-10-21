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
# 
from mpi4py import MPI

#nest to tvb
from transformer import store_data, analyse_data
#tvb to nest
from transformer import generate_data


# TODO: proper abstract class
# TODO: incoming (receive) and outgoing (send) 'pivot'
# TODO: analyse -> here transformer and science together -> split up

class NestTvbPivot():
    
    def __init(self, param):
        self.__param = param
        
        
    def _receive(self, comm_receiver, databuffer):
        '''
        Receive data on rank 0. Put it into the shared mem buffer.
        Replaces the former 'receive' function.
        NOTE: First refactored version -> not pretty, not final. 
        '''
        status_ = MPI.Status()
        num_sending = comm_receiver.Get_remote_size() # how many NEST ranks are sending?
        # TODO: It seems the 'check' variable is used to receive tags from NEST, i.e. ready for send...
        # change this in the future, also mentioned in the FatEndPoint solution from Wouter.
        check = np.empty(1,dtype='b')
        shape = np.empty(1, dtype='i')    
        count = 0
        # TODO: the last two buffer entries are used for shared information
        # --> they replace the status_data variable from previous version
        # --> find more elegant solution?
        databuffer[-1] = 1 # set buffer to 'ready to receive from nest'
        databuffer[-2] = 0 # marks the 'head' of the buffer
        
        while(True):
            head_ = 0 # head of the buffer, reset after each iteration
            
            # TODO: This is still not correct. We only check for the Tag of the last rank.
            # TODO: IF all ranks send always the same tag in one iteration (simulation step)
            # TODO: then this works. But it should be handled differently!!!!
            for i in range(num_sending):
                # new: We do not care which source sends first, give MPI the freedom to send in whichever order.
                comm_receiver.Recv([check, 1, MPI.CXX_BOOL], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status_)
            # TODO: handle properly, all ranks send tag 0?
            if status_.Get_tag() == 0:
                # wait until ready to receive new data (i.e. the sender has cleared the buffer)
                while databuffer[-1] != 1: # TODO: use MPI, remove the sleep
                    time.sleep(0.001)
                    pass
                for source in range(num_sending):
                    # send 'ready' to the nest rank
                    comm_receiver.Send([np.array(True,dtype='b'),MPI.BOOL],dest=source,tag=0)
                    # receive package size info
                    comm_receiver.Recv([shape, 1, MPI.INT], source=source, tag=0, status=status_)
                    # NEW: receive directly into the buffer
                    comm_receiver.Recv([databuffer[head_:], MPI.DOUBLE], source=source, tag=0, status=status_)
                    head_ += shape[0] # move head 
                    # TODO: revisit and check for proper encapsulation
                    # Here, storing and adding the spikes to the histogram was done
                    # Old code: store.add_spikes(count,data)
                    # This increased the workload of this MPI rank.
                    # All science and analysis stuff is moved to the 'sender' part. Because future parallel.
                # Mark as 'ready to do analysis'
                databuffer[-1] = 0
                # important: head_ is first buffer index WITHOUT data.
                databuffer[-2] = head_
            # TODO: handle properly, all ranks send tag 1?
            elif status_.Get_tag() == 1:
                count += 1
            # TODO: handle properly, all ranks send tag 2?
            elif status_.Get_tag() == 2:
                # end
                break
            else:
                raise Exception("bad mpi tag"+str(status_.Get_tag()))
        
    
    
    def _send(self, comm_sender, databuffer):
        '''
        Analysis/Science on INTRAcommunicator (multiple MPI ranks possible).
        TODO: not yet used, see also analysis function below
        Send data to TVB on INTERcommunicator comm_sender (multiple MPI ranks possible).
        Replaces the former 'send' function.
        NOTE: First refactored version -> not pretty, not final. 
        '''

        count=0
        status_ = MPI.Status()
        while True:
            # TODO: this communication has the 'rank 0' problem described in the beginning
            accept = False
            #logger.info("Nest to TVB : wait to send " )
            while not accept:
                req = comm_sender.irecv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG)
                accept = req.wait(status_)
            #logger.info(" Nest to TVB : send data status : " +str(status_.Get_tag()))
            if status_.Get_tag() == 0:
                # wait until the receiver has cleared the buffer, i.e. filled with new data
                while databuffer[-1] != 0: # TODO: use MPI, remove the sleep
                    time.sleep(0.001)
                    pass
                # TODO: All science/analysis here. Move to a proper place.
                times,data = _analyse(count, databuffer)
                
                # Mark as 'ready to receive next simulation step'
                databuffer[-1] = 1
                
                ############ OLD Code
                # TODO: this communication has the 'rank 0' problem described in the beginning
                #logger.info("Nest to TVB : send data :"+str(np.sum(data)) )
                # time of stating and ending step
                comm_sender.Send([times, MPI.DOUBLE], dest=status_.Get_source(), tag=0)
                # send the size of the rate
                size = np.array(int(data.shape[0]),dtype='i')
                comm_sender.Send([size,MPI.INT], dest=status_.Get_source(), tag=0)
                # send the rates
                comm_sender.Send([data,MPI.DOUBLE], dest=status_.Get_source(), tag=0)
                ############ OLD Code end
            elif status_.Get_tag() == 1:
                # disconnect when everything is ending
                break
            else:
                raise Exception("bad mpi tag"+str(status_.Get_tag()))
            count+=1
        #logger.info('NEST_to_TVB: End of send function')


    # See todo in the beginning, encapsulate I/O, transformer, science parts
    def _analyse(self, count, databuffer):
        '''
        All analysis and science stuff in one place.
        Done in three steps, that were previously disconnected.
        Step 1 and 2 were done in the receiving thread, step 3 in the sending thread.
        NOTE: All science and analysis is the same as before.
        :param count: Simulation iteration/step
        :param databuffer: The buffer contains the spikes of the current step
        :return times, data: simulation times and the calculated rates
        
        TODO: Step 1 and 2 can be merged into one step. Buffer is no longer filled rank by rank.
        TODO: Make this parallel with the INTRA communicator (should be embarrassingly parallel).
        '''
        
        # TODO: proper init and import
        #store: Python object, create the histogram 
        #analyse: Python object, calculate rates
        store = store_data(self.__param)
        analyse = analyse_data(self.__param)
        
        # Step 1) take all data from buffer and create histogram
        # second to last index in databuffer denotes how much data there is
        store.add_spikes(count,databuffer[:int(databuffer[-2])])
        # Step 2) take the resulting histogram
        data_to_analyse = store.return_data()
        # Step 3) Analyse this data, i.e. calculate rates?
        times,data = analyse.analyse(count,data_to_analyse)
        
        return times, data
        

class TvbNestPivot(): 
    
    def _receive(self, comm_receiver, databuffer):
        '''
        Receive data on rank 0. Put it into the shared mem buffer.
        Replaces the former 'receive' function.
        NOTE: First refactored version -> not pretty, not final. 
        '''
        status_ = MPI.Status()
        num_sending = comm_receiver.Get_remote_size() # how many TVB ranks are sending?
        # init placeholder for incoming data
        time_step = np.empty(2, dtype='d') # two doubles with start and end time of the step
        size = np.empty(1, dtype='i') # size of the rate-array
        # TODO: the last two buffer entries are used for shared information
        # --> they replace the status_data variable from previous version
        # --> find more elegant solution?
        databuffer[-1] = 1 # set buffer to 'ready to receive from tvb'
        databuffer[-2] = 0 # marks the 'head' of the buffer
        
        while True:
            # TODO: NEST to TVB transformer: irecv
            # TODO: TVB to NEST transformer (here): isend
            # TODO: --> rework communication protocol between simulators and transformers!
            requests=[]
            # logger.info(" TVB to Nest: wait receive ")
            for rank in range(num_sending):
                requests.append(comm_receiver.isend(True,dest=rank,tag=0))
            MPI.Request.Waitall(requests)
            # logger.info(" TVB to Nest: receive all")
            
            # TODO: works for now, needs rework if multiple ranks are used on TVB side
            # TODO: we receive from "ANY_SOURCE", but only check the status_ of the last receive...
            # get the starting and ending time of the simulation step
            # NEW: receive directly into the buffer
            comm_receiver.Recv([databuffer[0:], MPI.DOUBLE], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status_)
            # logger.info(" TVB to Nest: get time_step "+str(time_step)+" status : " + str(status_.Get_tag()))
            if status_.Get_tag() == 0:
                # wait until ready to receive new data (i.e. the sender has cleared the buffer)
                while databuffer[-1] != 1: # TODO: use MPI, remove the sleep
                    time.sleep(0.001)
                    pass
                # Get the size of the data
                comm_receiver.Recv([size, 1, MPI.INT], source=status_.Get_source(), tag=0, status=status_)
                # NEW: receive directly into the buffer
                # First two entries are the times, see above
                comm_receiver.Recv([databuffer[2:], MPI.DOUBLE], source=status_.Get_source(), tag=0, status=status_)
                # Mark as 'ready to do analysis'
                databuffer[-1] = 0
                databuffer[-2] = size # info about size of data array
                # logger.info(" TVB to Nest: update buffer")
            elif status_.Get_tag() == 1:
                # logger.info('TVB: end simulation')
                break
            else:
                raise Exception("bad mpi tag"+str(status_.Get_tag()))
        
        # logger.info('TVB_to_NEST: End of receive function')


    def _send(comm_sender, databuffer, generator, id_first_spike_detector):
        '''
        Generator/Science on INTRAcommunicator (multiple MPI ranks possible).
        TODO: not yet used.
        Send data to NEST on INTERcommunicator comm_sender (multiple MPI ranks possible).
        Replaces the former 'send' function.
        NOTE: First refactored version -> not pretty, not final. 
        
        TODO: Discuss communication protocol of TVB<->transformer and transformer<->NEST
        '''
        status_ = MPI.Status()
        num_sending = comm_sender.Get_remote_size() # how many TVB ranks are sending?
        # init placeholder for incoming data
        check = np.empty(1,dtype='b')
        size_list = np.empty(1, dtype='i')
        while(True):
            # TODO: This is still not correct. We only check for the Tag of the last rank.
            # TODO: IF all ranks send always the same tag in one iteration (simulation step)
            # TODO: then this works. But it should be handled differently!!!!
            for rank in range(num_sending):
                comm_sender.Recv([check, 1, MPI.CXX_BOOL], source=rank, tag=MPI.ANY_TAG, status=status_)
            # logger.info("TVB to NEST : send data status : " +str(status_.Get_tag()))
            # TODO: handle properly, all ranks send tag 0?
            if status_.Get_tag() == 0:
                # wait until the receiver has cleared the buffer, i.e. filled with new data
                while databuffer[-1] != 0: # TODO: use MPI, remove the sleep
                    time.sleep(0.001)
                    pass

                # TODO: All science/generate here. Move to a proper place.
                # method: generate_spike(count,time_step,rate)
                # NOTE: count is a hardcoded '0'. Why?
                # NOTE: time_step are the first two doubles in the buffer
                # NOTE: rate is a double array, which size is stored in the second to last index
                generator = generate_data(path_config+'/../../log/',nb_spike_generator,param)
                spikes_times = generator.generate_spike(0,databuffer[:2],databuffer[2:int(databuffer[-2])])
                # logger.info(" TVB to Nest: spike time")
                
                # Mark as 'ready to receive next simulation step'
                databuffer[-1] = 1
                
                ###### OLD code, kept the communication and science as it is for now
                ### TODO: Receive from status_.Get_source() and rank
                ### TODO: Send to status_.Get_source() and rank
                ### TODO: why???
                ### TODO: a second status_ object is used, should not be named the same
                for rank in range(num_sending):
                    # NOTE: in 'test_receive_tvb_to_nest.py': hardcoded 10
                    comm_sender.Recv([size_list, 1, MPI.INT], source=rank, tag=0, status=status_)
                    if size_list[0] != 0:
                        list_id = np.empty(size_list, dtype='i')
                        # NOTE: in 'test_receive_tvb_to_nest.py': hardcoded np.arange(0,10,1)
                        comm_sender.Recv([list_id, size_list, MPI.INT], source=status_.Get_source(), tag=0, status=status_)
                        # Select the good spike train and send it
                        # logger.info(" TVB to Nest:"+str(data))
                        # logger.info("rank "+str(rank)+" list_id "+str(list_id))
                        # TODO: Creating empty lists and append to them in a loop, all inside a loop
                        # TODO: this is slow and will be a bottleneck when we scale up.
                        data = []
                        shape = []
                        for i in list_id:
                            shape += [spikes_times[i-id_first_spike_detector].shape[0]]
                            data += [spikes_times[i-id_first_spike_detector]]
                        send_shape = np.array(np.concatenate(([np.sum(shape)],shape)), dtype='i')
                        # firstly send the size of the spikes train
                        comm_sender.Send([send_shape, MPI.INT], dest=status_.Get_source(), tag=list_id[0])
                        # secondly send the spikes train
                        data = np.concatenate(data).astype('d')
                        comm_sender.Send([data, MPI.DOUBLE], dest=rank, tag=list_id[0])
                # logger.info(" end sending:")
                ###### OLD code end
            elif  status_.Get_tag() == 1:
                # logger.info(" TVB to Nest end sending") # NOTE: one sim step?
            elif status_.Get_tag() == 2:
                # logger.info(" TVB to Nest end simulation ") # NOTE: end whole sim.
                break
            else:
                raise Exception("bad mpi tag : "+str(status_.Get_tag()))
        
        # logger.info('TVB_to_NEST: End of send function')


    # See todo in the beginning, encapsulate I/O, transformer, science parts
    def _analyse(self):
        pass
