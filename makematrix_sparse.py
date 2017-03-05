#make matrix rep of hamiltonan
import numpy as np
from scipy.sparse import lil_matrix
import itertools

def Braket(basisseq,column,row,spin):
    braket=np.hstack((np.fliplr(basisseq[row])[spin],basisseq[column,spin]))
    return braket

def Parity(sitesize,diff,braket):# fermion sign
    #site number of an electron ani
    anidex=np.asscalar(np.array((np.where(diff==1))))
    #site number of electron cre
    cridex=np.asscalar(np.array((np.where(diff==-1))))
    overelecnum=sum(braket[sitesize-cridex:sitesize+anidex])
    if overelecnum%2==0:#whether the number of overjump cre op. is even or not
        parity=1
    else:
        parity=-1
    return parity

def  MatrixRep(sitesize,basisseq,basissize,t1,t2,U,V1,V2,bc):
    MR=lil_matrix((basissize,basissize))#initialization
    for i, j in itertools.product(range(basissize), range(basissize)):
    #for i in range(basissize):
    #for j in range(basissize):
        Diff =basisseq[i]-basisseq[j]
        if (Diff[0]==0).all():
            if (Diff[1]==0).all():#diagonal elements
                MR[i,j]+=U*np.array(np.where(sum(basisseq[i],0)==2)).size#onsite interaction
                for l in range(sitesize-1):#NNinteraction(nonBoundary)
                    MR[i,j]+=V1*sum(basisseq[i],0)[l]*sum(basisseq[i],0)[l+1]
                MR[i,j]+=V1*sum(basisseq[i],0)[sitesize-1]*sum(basisseq[i],0)[0]*bc*bc#NNinteraction(boundary)
                for m in range(sitesize-2):#NNNinteraction(nonboundary)
                    MR[i,j]+=V2*sum(basisseq[i],0)[m]*sum(basisseq[i],0)[m+2]
                MR[i,j]+=V2*sum(basisseq[i],0)[sitesize-2]*sum(basisseq[i],0)[0]*bc*bc#NNNinteraction(boundary)1
                MR[i,j]+=V2*sum(basisseq[i],0)[sitesize-1]*sum(basisseq[i],0)[1]*bc*bc#NNNinteraction(boundary)2
            else: #downhopping
                if np.array(np.nonzero(Diff[1])).size==2:
                #exclude many electrons hopping concurrently
                    if abs(np.array(np.nonzero(Diff[1]))[0,0]-np.array(np.nonzero(Diff[1]))[0,1])==1:
                        #NNhopping(NonBoundary)
                        bk=Braket(basisseq,i,j,1)
                        MR[i,j]+=-t1*Parity(sitesize,Diff[1],bk)
                    elif abs(np.array(np.nonzero(Diff[1]))[0,0]-np.array(np.nonzero(Diff[1]))[0,1])==2:
                        #NNNhopping(Nonboundary)
                        bk=Braket(basisseq,i,j,1)
                        MR[i,j]+=-t2*Parity(sitesize,Diff[1],bk)
                    if abs(np.array(np.nonzero(Diff[1]))[0,0]-np.array(np.nonzero(Diff[1]))[0,1])==sitesize-1:
                        #NNhopping(Boundary)
                        bk=Braket(basisseq,i,j,1)
                        MR[i,j]+=-t1*bc*Parity(sitesize,Diff[1],bk)
                    elif abs(np.array(np.nonzero(Diff[1]))[0,0]-np.array(np.nonzero(Diff[1]))[0,1])==sitesize-2:
                        #NNNhopping(Boundary)
                        bk=Braket(basisseq,i,j,1)
                        MR[i,j]+=-t2*bc*Parity(sitesize,Diff[1],bk)
        else:
            if (Diff[1]==0).all():#uphopping
                if np.array(np.nonzero(Diff[0])).size==2:
                #exclude many electrons hopping concurrently
                    if abs(np.array(np.nonzero(Diff[0]))[0,0]-np.array(np.nonzero(Diff[0]))[0,1])==1:
                        #NNhopping(NonBoundary)
                        bk=Braket(basisseq,i,j,0)
                        MR[i,j]+=-t1*Parity(sitesize,Diff[0],bk)
                    elif abs(np.array(np.nonzero(Diff[0]))[0,0]-np.array(np.nonzero(Diff[0]))[0,1])==2:
                        #NNNhopping(Nonboundary)
                        bk=Braket(basisseq,i,j,0)
                        MR[i,j]+=-t2*Parity(sitesize,Diff[0],bk)
                    if abs(np.array(np.nonzero(Diff[0]))[0,0]-np.array(np.nonzero(Diff[0]))[0,1])==sitesize-1:
                        #NNhopping(Boundary)
                        bk=Braket(basisseq,i,j,0)
                        MR[i,j]+=-t1*bc*Parity(sitesize,Diff[0],bk)
                    elif abs(np.array(np.nonzero(Diff[0]))[0,0]-np.array(np.nonzero(Diff[0]))[0,1])==sitesize-2:
                        #NNNhopping(Boundary)
                        bk=Braket(basisseq,i,j,0)
                        MR[i,j]+=-t2*bc*Parity(sitesize,Diff[0],bk)
    MR=MR.tocsr()
    return MR

if __name__ == '__main__':
    import makebasis as mb
    hop1=10#NN hopping
    hop2=1#NNN hopping
    U=100#onsite interaction
    Vint1=10#NN interaction
    Vint2=1#NNN interaction
    BC=1#coeffiient of boundary conditon(PBC＝１,APBC=-1,OBC=0)
    SiteSize=8
    ElecSize=4
    TotalSpin=0
    BasisSeq,BasisSize=mb.BasisSeqSize(SiteSize,ElecSize,TotalSpin)
    H=MatrixRep(SiteSize,BasisSeq,BasisSize,hop1,hop2,U,Vint1,Vint2,BC)
