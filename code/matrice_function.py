#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mohamed NIANG
"""

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


nbligne=10**3
nbcol=10**4

BLOCKDIMx=16
BLOCKDIMy=16
GRIDDIMx=nbcol//BLOCKDIMx+1
GRIDDIMy=nbligne//BLOCKDIMy+1

a=np.zeros( [nbligne,nbcol],dtype=np.float32)
for i in range (nbligne):
    for j in range (nbcol):
        a[i,j]=i/nbligne*10
b=np.empty_like(a)

f = open("mat_func.cu", 'r')
cuda_source = "".join(f.readlines())
mod=SourceModule(cuda_source)
applique_func=mod.get_function("applique_fonc")


input_gpu=cuda.mem_alloc(a.nbytes)
output_gpu=cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(input_gpu,a)
applique_func(np.int32(nbligne), np.int32(nbcol), input_gpu, output_gpu, block=(BLOCKDIMx,BLOCKDIMy,1), grid=(GRIDDIMx,GRIDDIMy,1))
cuda.memcpy_dtoh(b,output_gpu)
