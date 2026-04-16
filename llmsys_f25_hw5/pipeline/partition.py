from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast
import torch
from torch import nn

class WithDevice(nn.Module):
    def __init__(self, module: nn.Module, device: torch.device):
        super().__init__()
        self._module = module
        self._device = torch.device(device)

    def forward(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    @property
    def module(self):
        return self._module

    @property
    def device(self):
        return self._device

def _retrieve_device(module: nn.Module) -> torch.device:
    device = None
    for parameter in module.parameters():
        if device is None:
            device = parameter.device
        elif device != parameter.device:
            raise ValueError(
                f'nn.Module: {module}, should have all parameters on a single device,'
                ' please use .to() to place the module on a single device')

    return device if device is not None else torch.device("cpu")

def _assemble_partition(modules: List[nn.Module]):
    modules_list: List[nn.Module] = []
    for module in modules:
        if isinstance(module, nn.Sequential):
            modules_list.extend(module.children())
        else:
            modules_list.append(module)
    return nn.Sequential(*modules_list)

def _split_module(modules: nn.Sequential) -> Tuple[List[nn.Sequential], List[torch.device]]:
    '''Split an nn.Sequential module into partitions and devices.

    Each partition is a nn.Sequential module attached to the same device.
    The partitions and devices are returned as a tuple. Each partition corresponds to a device in the devices list.
    
    Hint: 
    1. You can use the _retrieve_device function to retrieve the device of a module.
    2. However, users might use the WithDevice class to wrap a module with a device. In this case, you should use the device from the WithDevice class.
    3. You can use the _assemble_partition function to assemble a partition from a list of modules.
    '''
    partitions = []
    devices = []

    current_partition = []
    current_device = None
    for name, module in modules.named_children():
        # BEGIN_HW5_2_1
        
        #如果用户用WithDevice封装了某个Module 这时候不能用_retrieve_device
        if isinstance(module, WithDevice):
            new_device=module.device
        else:
            new_device=_retrieve_device(module)
            
        #如果是输入的第一个
        if current_device==None:
            current_device=new_device
        
        #如果上一个device(current_device)和当前的device(new_device)一样
        if current_device==new_device:
            current_partition.append(module)
        else:
            #如果不一样 则说明设备发生了改变 先把当前的partition提交 然后重新开一个partition
            partitions.append(_assemble_partition(current_partition))
            devices.append(current_device)
            #初始化current_partition 然后再把当前的module放进去
            current_partition=[]
            current_partition.append(module)
            current_device=new_device
        
        
        # END_HW5_2_1

    if current_device is not None:
        partitions.append(_assemble_partition(current_partition))
        devices.append(current_device)

    partitions = nn.ModuleList(partitions)

    return partitions, devices
