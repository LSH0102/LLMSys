from random import Random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


class Partition():
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        '''Given index, get the data according to the partitioned index'''
        # BEGIN_HW5_1_1
        return self.data[self.index[index]]
        # END_HW5_1_1

class DataPartitioner():
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        ''' Create indices for different partitions
        1. Create indices and use `rng` to shuffle indices
        2. Create different partitions of indices according to `sizes` and store in `self.partitions`
        '''
        # BEGIN_HW5_1_1
        index=[i for i in range(0,len(data))]
        
        #原地修改列表 返回None 参考random.py的用法
        rng.shuffle(index)
        
        last=0
        for i in range(0,len(sizes)):
            curr=last+int(sizes[i]*len(index))
            self.partitions.append(index[last:curr])
            last=curr
            
        # END_HW5_1_1

    def use(self, partition):
        ''' Return a simple dataset class `Partiton` by original data and partitioned indices

        Just one line of code. Think it simply.
        '''
        # BEGIN_HW5_1_1
        return Partition(self.data, partition)
        # END_HW5_1_1

def partition_dataset(rank, world_size, dataset, batch_size=128, collate_fn=None):
    """ Partitioning training dataset of the Machine Translation

    Returns:
        DataLoader: partitioned dataloader
    
    Hint:
    1. Calculate the partitioned batch size
    2. Create a partitioner class `DataPartitioner` with dataset and the list of partitioned sizes
    3. Get the current partition dataset given `rank`, use the `use` function in DataPartitioner
    4. Wrap the dataset with `DataLoader`, remember to customize the `collate_fn`
    """
    # BEGIN_HW5_1
    n=len(dataset)//world_size
    ratio_list=world_size*[1.0/world_size ]
    
    partitioner=DataPartitioner(dataset,ratio_list)
    
    P=partitioner.use(partitioner.partitions[rank])
    return DataLoader(P,batch_size=batch_size ,collate_fn=collate_fn)

    # END_HW5_1
