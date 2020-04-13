#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mohamed NIANG
"""

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

f = open("vec_sum.cu", 'r')
cuda_source = "".join(f.readlines())
mod=SourceModule(cuda_source)
vecteur_somme_kernel=mod.get_function("somme")

BLOCKDIM=512

n=10**6

a=np.arange(0,1,1/n,dtype=np.float32)
b=np.ones_like(a)
c=np.empty_like(a)

input1_gpu=cuda.mem_alloc(a.nbytes)
input2_gpu=cuda.mem_alloc(a.nbytes)
output_gpu=cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(input1_gpu,a)
cuda.memcpy_htod(input2_gpu,b)
vecteur_somme_kernel(np.int32(a.size), input1_gpu, input2_gpu, output_gpu ,block=(BLOCKDIM,1,1), grid=(a.size//BLOCKDIM+1,1,1))
cuda.memcpy_dtoh(c,output_gpu)

