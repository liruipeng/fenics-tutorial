import numpy as np
from numpy import linalg as LA
import os
import pdb

def readtxt(Nx, Ny, fn):
   f = open(fn, 'r')
   data = []
   for line in f:
      tokens = line.split()
      f_tokens = [float(x) for x in tokens]
      if Ny is None:
         curr_data = np.array(f_tokens).reshape((1,Nx))
      else:
         curr_data = np.array(f_tokens).reshape((1,Nx,Ny))
      data.append(curr_data)
   f.close()
   data_array = np.concatenate(data, axis=0)
   return data_array

if __name__ == '__main__':
   nx = 32
   dirpath='data3/'
   # input
   Kappa = readtxt(Nx=nx, Ny=nx, fn=os.path.join(dirpath,'Kappa.txt'))
   F = readtxt(Nx=nx, Ny=nx, fn=os.path.join(dirpath,'F.txt'))
   U = readtxt(Nx=nx, Ny=nx, fn=os.path.join(dirpath,'U.txt'))
   COO = readtxt(Nx=2, Ny=None, fn=os.path.join(dirpath,'coords.txt'))
   X = COO[1:,0].reshape(32,32)
   Y = COO[1:,1].reshape(32,32)
   #pdb.set_trace()
   # output
   np.savez(os.path.join(dirpath,'Poisson3.npz'), Kappa=Kappa, F=F, U=U, X=X, Y=Y)

