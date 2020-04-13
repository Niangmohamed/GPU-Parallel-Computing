#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mohamed NIANG
"""
import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

nb_tirage=10000
BLOCKDIM=512
NB_BLOCK=12800

N=BLOCKDIM*NB_BLOCK
res=np.empty(N).astype(np.int32)

f = open("pi_kernel.cu", 'r')
cuda_source = "".join(f.readlines())
# compile et recupere la fonction
mod=SourceModule(cuda_source,no_extern_c=True)
counthits=mod.get_function('counthits')


res_gpu=cuda.mem_alloc(res.nbytes)
t1=time.time()
counthits(np.int32(nb_tirage), res_gpu,np.int32(0), block=(BLOCKDIM,1,1), grid=(NB_BLOCK,1))
cuda.memcpy_dtoh(res,res_gpu)
t2=time.time()
somme=sum(res/(nb_tirage))


print("Estimation de pi basée sur ", 2*N*nb_tirage/10**9, "*10**9 tirages :", somme/N*4)
print("Temps :", t2-t1, "secondes.")