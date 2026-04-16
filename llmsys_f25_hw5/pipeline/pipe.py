from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN_HW5_2_1
    res=[ [] for i in range(0,num_batches+num_partitions-1)]
    for i in range(0,num_batches):
        for j in range(0,num_partitions):
            res[i+j].append(tuple([i,j]))
    return res
    # END_HW5_2_1

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN_HW5_2_2
        if x.shape[0]<self.split_size:
            micro_batches=[x]
            
            schedule=_clock_cycles(1, len(self.partitions))
        else:
            
            micro_batches=list(torch.split(x,self.split_size,dim=0))
            
            schedule=_clock_cycles(x.shape[0]//self.split_size, len(self.partitions))
        
        for i in range(0,len(schedule)):
            curr_sche=schedule[i]
            self.compute(micro_batches, curr_sche)
        
        res=[]
        for item in micro_batches:
            res.append(item.to(self.devices[-1]))
        
        res=torch.concat(res,dim=0)
        return res
        # END_HW5_2_2

    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN_HW5_2_2
        for t in schedule:
            partition=partitions[t[1]]
            device=devices[t[1]]
            #split_size就是数据集的分割大小
            
            b=batches[t[0]].to(device)
            
            class functor:
                def __init__(self,module,input):
                    self.module=module
                    self.input=input
                
                def __call__(self,):
                    return self.module(self.input)
                    
            
            task=Task(functor(partition,b))
            
            self.in_queues[t[1]].put(task)
            
        for t in schedule:
            res=self.out_queues[t[1]].get()
            
            batches[t[0]]=res[-1][-1]
        # END_HW5_2_2
