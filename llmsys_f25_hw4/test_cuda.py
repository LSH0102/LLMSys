# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 13:30:32 2025

@author: 25488
"""

import torch
import numpy as np

x=np.random.normal(size=(150,15000))

x=torch.Tensor(x).float()
x=x.to('cuda')
print(x)
print(torch.allclose(x.flatten(),x.flatten()))