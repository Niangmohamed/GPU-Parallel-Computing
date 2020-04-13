#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mohamed NIANG
"""

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
import matplotlib.animation as animation


nbligne=1024
nbcol=1024

BLOCKDIMx=16
BLOCKDIMy=16
GRIDDIMx=nbcol//BLOCKDIMx+1
GRIDDIMy=nbligne//BLOCKDIMy+1

a=np.zeros( [nbligne,nbcol],dtype=np.float32)

f = open("mat_func.cu", 'r')
cuda_source = "".join(f.readlines())
mod=SourceModule(cuda_source)
mandel=mod.get_function("mandelbrot")


output_gpu=cuda.mem_alloc(a.nbytes)
mandel(np.int32(nbligne), np.int32(nbcol), np.float32(4), np.float32(-2), np.float32(2), np.float32(-2), np.float32(2), output_gpu, block=(BLOCKDIMx,BLOCKDIMy,1), grid=(GRIDDIMx,GRIDDIMy,1))
cuda.memcpy_dtoh(a,output_gpu)

plt.imshow(a)

fig=plt.figure()
b=np.zeros( [nbligne,nbcol],dtype=np.float32)
ims = []
def create_anim():
  for seuil in np.arange(0.2,4,0.01):
     mandel(np.int32(nbligne), np.int32(nbcol), np.float32(seuil), np.float32(-2), np.float32(2), np.float32(-2), np.float32(2), output_gpu, block=(BLOCKDIMx,BLOCKDIMy,1), grid=(GRIDDIMx,GRIDDIMy,1))
     cuda.memcpy_dtoh(b,output_gpu)
     im = plt.imshow(b, animated=True)
     ims.append([im])

# lancer la fonction ci dessus pour creer l'animation et decommenter ci dessous pour la voir
create_anim()
ani = animation.ArtistAnimation(fig, ims, interval=5, blit=True)
plt.show()
