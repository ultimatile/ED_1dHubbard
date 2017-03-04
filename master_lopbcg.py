#main ED code 
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as ssp
import makebasis as mb
import makematrix_sparse as mm
from time import time
from scipy.sparse.linalg import lobpcg
from scipy import rand

hop1=1#NN hopping
hop2=0#NNN hopping
U=10#onsite interaction
Vint1=4#NN interaction
Vint2=2#NNN interaction
BC=-1#coeffient of boundary condition(PBC＝１,APBC=-1,OBC=0)
SiteSize=8
ElecSize=4
TotalSpin=0

#make basis
start=time()
BasisSeq,BasisSize=mb.BasisSeqSize(SiteSize,ElecSize,TotalSpin)
#make matrix rep.
A=mm.MatrixRep(SiteSize,BasisSeq,BasisSize,hop1,hop2,U,Vint1,Vint2,BC)
#diagonalization
x = rand(BasisSize,1)#intial random vectors
#rand's 2nd arg is the same number of target eigenstates.
Evalue,Evec=lobpcg(A,x,A-2*ssp.identity(BasisSize),largest=False)
#lobpcg's 3rd arg is a preconditioner whose defalut is identity.
print(time()-start)
print(Evalue)
print(min(Evalue),np.where(Evalue==min(Evalue)))# GS energy
