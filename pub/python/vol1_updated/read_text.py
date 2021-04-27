import numpy as np
from numpy import linalg as LA

import pdb

def readtxt(N, fn):
   f = open(fn, 'r')
   data = []
   for line in f:
      tokens = line.split(' ')
      f_tokens = [float(x) for x in tokens]
      curr_data = np.array(f_tokens).reshape((1,N,N))
      data.append(curr_data)
   f.close()
   data_array = np.concatenate(data, axis=0)
   return data_array

if __name__ == '__main__':
   Kappa = readtxt(N=32, fn="data1/Kappa.txt")
   F = readtxt(N=32, fn="data1/F.txt")
   U = readtxt(N=32, fn="data1/U.txt")
   np.savez("data1/Poisson1.npz", Kappa=Kappa, F=F, U=U)

