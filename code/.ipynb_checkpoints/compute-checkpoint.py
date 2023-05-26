import pandas as pd
import numpy as np
import csv 
import json
import os
from sklearn.utils.extmath import randomized_svd
import scipy.stats as stats
from numpy.random import RandomState,SeedSequence
from numpy.random import MT19937
from scipy import signal

def alignement(df,  offset_TR, onset_TR=72, affiche=False):
    if(affiche):print(f"Number of scans :{len(df)} \n")
    df.drop(range(0,onset_TR), axis=0, inplace=True)
    if(affiche):print(f"Number of scans after onset removing :{len(df)} \n")
    df.drop(range(offset_TR, len(df)), inplace=True)
    if(affiche):print(f"Number of scans after offset removing:{len(df)} \n")
    df.reset_index(drop=True, inplace=True)
    
    return df

def resampling(annot_data):
    TRdur13 = round(np.shape(annot_data)[0]/1.3)
    annot_res = pd.DataFrame(signal.resample(annot_data,TRdur13), columns=annot_data.columns)
    return annot_res



def normed(X, axis=0):
    
    ## Get matrix norm 
    normed = np.array(X)
    normal_base = np.linalg.norm(normed, axis=axis, keepdims=True)

    
    
    ## Avoid DivideByZero errors
    zero_items = np.where(normal_base == 0)
    normal_base[zero_items] = 1
    
    ## Normalize and re-set zero_items to 0
    normed = normed / normal_base
    normed[zero_items] = 0
   

    return normed

def standarization (X,Y):
    X_normed=pd.DataFrame(normed(X.apply(stats.zscore)), columns=X.columns)
    Y_normed=pd.DataFrame(normed(Y.apply(stats.zscore)), columns=Y.columns)
    return X_normed, Y_normed

def R_cov(X, Y) : 
    
    #X, Y should be Array
    R = (Y.T @ X)
    
    return R


def SVD(R, ICA=False,n_components=None, seed=1):
    R = np.array(R)
    n_components = min(R.shape)
    
    # Run most computationally efficient SVD
    U, d, V = randomized_svd(R, n_components=n_components, random_state=seed)
    V=V.T

    result = np.where(np.abs(V)==np.amax(np.abs(V), axis=0))
    if(ICA):
        for i in range(len(result[0])):
            if(np.sign(V[result[0][i],result[1][i]]))<0 : 
                V[:,result[1][i]]=-V[:,result[1][i]]
                U[:,result[1][i]]=-U[:,result[1][i]]

    
    return pd.DataFrame(U), np.diag(d), pd.DataFrame(V)


def varexp(singular):
    if singular.ndim != 2:
        raise ValueError('Provided `singular` array must be a square diagonal '
                         'matrix, not array of shape {}'
                         .format(singular.shape))
    return (np.diag(singular)**2 / np.sum(np.diag(singular)**2))

def PLS_scores(X, Y, U, V):
    Lx= X@np.array(V)
    Ly= Y@np.array(U)
    return Lx, Ly

def PLS_loadings (X,Y, Lx,Ly): 
    corr_Lx_X= pd.concat([Lx, X], axis=1, keys=['df1', 'df2']).corr().sort_index().loc['df2', 'df1']
    corr_Lx_Y= pd.concat([Lx, Y.sort_index()], axis=1, keys=['df1', 'df2']).corr().sort_index().loc['df2', 'df1']
    corr_Ly_Y= pd.concat([Ly, Y.sort_index()], axis=1, keys=['df1', 'df2']).corr().sort_index().loc['df2', 'df1']
    corr_Ly_X= pd.concat([Ly, X.sort_index()], axis=1, keys=['df1', 'df2']).corr().sort_index().loc['df2', 'df1']
    return corr_Lx_X,corr_Lx_Y,corr_Ly_X,corr_Ly_Y
        
def rotatemat (origlv, bootlv):
    #define coordinate space between original and bootstrap LVs
    tmp=origlv.T@bootlv
    

    #orthogonalze space
    [V,W,U]=SVD(tmp);
   
    #determine procrustean transform
    new_mat=U@V.T

    return new_mat

def permu(X,Y,U,nPerms, seed=1):
    
    Sp_new=[]
    
    # Check that dimensions of X & Y are correct
    if(X.shape[0] != Y.shape[0]): raise Exception("Input arguments X and Y should have the same number of rows")
    
    rs = RandomState(MT19937(SeedSequence(seed)))
    for i in range(nPerms):
      

        Xp=X
       
        
        # Permute Y rows 
        Yp=Y.sample(frac=1,replace=False, random_state=rs)
    
    
        # Generate cross-covariance matrix between X and permuted Y
        Rp = R_cov(np.array(Xp), np.array(Yp))
        
       
        
        # Singular value decomposition of Rp
        Up, Sp, Vp = SVD(Rp)
       
    
        
        rot_mat=rotatemat(U,Up)
    
        Up = Up @ Sp @ rot_mat
        
        Sp = (np.sqrt(np.sum(np.square(Up), axis = 0)))
       
       
        Sp_new.append(Sp)
        
 
    return np.array(Sp_new).T

def myPLS_get_LC_pvals(Sp_vect,S,nPerms, seuil=0.01 ) : 
    
    sig_PLC=[]
    
    S_mat=np.repeat(np.diag(S), nPerms).reshape(len(np.diag(S)), nPerms)
   
    sp=np.sum(Sp_vect>=S_mat, axis=1)
   
    
    #Compute p-val
    LC_pvals = (sp+1)/(nPerms+1)
    signif_LC = np.argwhere(LC_pvals<seuil)
    nSignifLC = signif_LC.shape[0]
   
    for i in range(nSignifLC):
       
        sig_PLC.append(signif_LC[i])
        print(f"LC {sig_PLC[-1]} with p-value = {LC_pvals[sig_PLC[-1]]} \n")
        
    
    return LC_pvals, sig_PLC

def bootstats(vect):
    #Behavior/design and imaging saliences
    mean_=vect.mean(axis=(1,2))
    std_=vect.std(axis=(1,2))
    lB_=np.percentile(vect,2.5,axis=(1,2))
    uB_=np.percentile(vect,97.5,axis=(1,2))
    return mean_, std_, lB_, uB_


def myPLS_bootstrapping(X0,Y0,U,V, nBoots,  seed):
    """
    Input : 
    - X0 : NxM brain data (not normalized!)
    - Y0 : NxB  behavior/design data (not normalized!)
    - U : BxL Behavorial Saliences (LC)
    - V : BxL images Salience
    - nBoots : number of bootstrap sample 
    
    Output : 
    - Ub_vect : BxSxP : S #Lcs, P #boostrap sample, bootstrapped behavior saliences for all LCs
    - Vb_vect : MxSxP : bootstrapped brain saliences for all LCs
    - Lxb,.Lyb,.LC_img_loadings_boot,.LC_behav_loadings_boot :bootstrapping scores (see myPLS_get_PLSscores for details) 
    - mean : mean of boostraping disitrbution 
    - std : standard deviation of bootstrapping distributions
    - lB : lower bound of 95% confidence interval of bootstrapping distributions
    - uB : upper bound of 95% confidence interval of bootstrapping distributions
"""
    
    if(X0.shape[0] != Y0.shape[0]): raise Exception("Input arguments X and Y should have the same number of rows")
    rs = RandomState(MT19937(SeedSequence(seed)))
    
    
    Ub_vect=[]
    Vb_vect=[]
    
    boot_Lxb=[]
    boot_Lyb=[]
    boot_LC_img_loadings=[]
    boot_LC_behav_loadings=[]
    
    boot_results={}
    boot_stat={}
    
    for i in range(nBoots):
        Xb= X0.sample(frac=1, replace=True, random_state=rs)
       
        Yb = Y0.sample(frac=1, replace=True, random_state=rs)
        Xb, Yb = standarization(Xb, Yb)
        
        #Cross-covariance
        Rb = R_cov(Xb,Yb)
      
        
        #SVD 
        Ub, Sb, Vb = SVD(Rb)
        
        #Procrustas transform (correction for axis rotation/reflection), mode 2, average rotation of U and V
        #Computed on both U and V
        rotatemat1 = rotatemat(U, Ub)
        rotatemat2 = rotatemat(V, Vb)
        
        #Full rotation
        rotatemat_full = (rotatemat1 + rotatemat2)/2;
        
        Vb = Vb @ rotatemat_full
        Ub = Ub @ rotatemat_full
        
        Ub_vect.append(Ub)
        Vb_vect.append(Vb)
        
        Lxb, Lyb = PLS_scores(Xb, Yb,Ub, Vb)
        corr_Lxb_Xb,corr_Lyb_Yb,corr_Lxb_Yb,corr_Lyb_Xb = PLS_loadings(Xb, Yb, Lxb, Lyb)
        
        boot_Lxb.append(Lxb)
        boot_Lyb.append(Lyb)
        boot_LC_img_loadings.append(corr_Lxb_Xb)
        boot_LC_behav_loadings.append(corr_Lyb_Yb)
        
    boot_results['Ub_vect']=np.array(Ub_vect).T
    boot_results['Vb_vect']=np.array(Vb_vect).T
    boot_results['Lxb']=np.array(boot_Lxb).T
    boot_results['Lyb']=np.array(boot_Lyb).T
    boot_results['LC_img_loadings']=np.array(boot_LC_img_loadings).T
    boot_results['LC_behav_loadings']=np.array(boot_LC_behav_loadings).T
    
    boot_stat['Ub_mean'],boot_stat['Ub_std'], boot_stat['Ub_lB'], boot_stat['Ub_uB'] = bootstats(np.array(Ub_vect))
    boot_stat['Vb_mean'],boot_stat['Vb_std'], boot_stat['Vb_lB'], boot_stat['Vb_uB'] = bootstats(np.array(Vb_vect))
    boot_stat['LC_behav_mean'],boot_stat['LC_behav_std'], boot_stat['LC_behav_lB'], boot_stat['LC_behav_uB'] = bootstats(boot_results['LC_img_loadings'])
    boot_stat['LC_img_mean'],boot_stat['LC_img_std'], boot_stat['LC_img_lB'], boot_stat['LC_img_uB'] = bootstats(boot_results['LC_behav_loadings'])
    
    return boot_results, boot_stat