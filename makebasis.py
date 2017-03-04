#make basis rep for matrix rep
import numpy as np

def BasisSeqSize(sitesize,elecsize,totalspin):#return basis and its size
    #0th row:number of upspin
    #1st row ;number of downspin
    upseq=np.eye(sitesize,2*sitesize,dtype=int).reshape(sitesize,2,sitesize)
    downseq=np.eye(sitesize,2*sitesize,dtype=int,k=sitesize).reshape(sitesize,2,sitesize)
    oneeleseq=np.vstack((upseq,downseq))
    basisseq=oneeleseq#initialization
    for ElecNum in range(elecsize-1):
        basisseq=np.array([basisseq[i]+oneeleseq[j] for j in range(2*sitesize) for i in range(len(basisseq))])
    #delete  unnecessary  basis
    #delete basis considering Pauli's principle
    basisseq=np.delete(basisseq,np.where(basisseq>1),0)
    #delete basis whose spinz doesn't satisfy totalspin condition.
    basisseq=np.delete(basisseq,np.where(np.sum(basisseq[:,0]-basisseq[:,1],axis=1)!=totalspin),0)
    #delete duplicate basis
    dtype1 = np.dtype((np.void, basisseq.dtype.itemsize * np.prod(basisseq.shape[1:])))
    b = np.ascontiguousarray(basisseq.reshape(basisseq.shape[0],-1)).view(dtype1)
    basisseq=np.flipud(basisseq[np.unique(b, return_index=1)[1]])
    #this seq is ascending order, so here is arrangeing in descending order by flipud
    basissize=len(basisseq)
    return basisseq,basissize

if __name__ =='__main__':
    SiteSize=8
    ElecSize=4
    TotalSpin=0
    BasisSeq,BasisSize=BasisSeqSize(SiteSize,ElecSize,TotalSpin)
    print(BasisSize)
