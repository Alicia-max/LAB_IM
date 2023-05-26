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
from sklearn.utils.validation import check_X_y

def alignement(df,  offset_TR, onset_TR=72, affiche=False):
    """
    Aligns the fMRI to a specific start and offset 
    to return only signals measured during movies watching only 

    Input 
    -------
        - df (DataFrame) : voxel-wise series 
        - offset_TR (int) : specific time (in TR) corresponding to the end of the film 
        - Onset_TR (int): specific time (in TR) corresponding to the beginning of the film
        - affiche (Bool) : Determine if the number of scans removed is displayed
    Output 
    -------
        - df (DataFrame)
    """
    if(affiche):print(f"Number of scans :{len(df)} \n")
        
    ## Remove scans before movie wieving
    df.drop(range(0,onset_TR), axis=0, inplace=True)
    if(affiche):print(f"Number of scans after onset removing :{len(df)} \n")
        
    ## Remove scans after movie wieving    
    df.drop(range(offset_TR, len(df)), inplace=True)
    if(affiche):print(f"Number of scans after offset removing:{len(df)} \n")
       
    df.reset_index(drop=True, inplace=True)
    
    return df

def resampling(df, dur, resol=1.3):
    """
    Resamples and aligns the behavioral dataset to match the corresponding scan sets 
    
    Input 
    -------
        - df (DataFrame) : Behavorial Dataset
        - dur (int) : duration of the film
        - resol (float) : resolution of the fMRI dataset 
    Output
    -------
        - df (DataFrame) : behavioral dataset with the same resolution and size as the scans (fMRI) dataset
    
    """
    
    TRdur13 = round(np.shape(df)[0]/resol)
    ## Resampling
    df = pd.DataFrame(signal.resample(df,TRdur13), columns=df.columns)
    
    ## Alignement
    df.drop(range(dur-1, len(df)), axis=0, inplace=True)
    return df

def standarization (X,Y, type_="zscore", durations=None):
    """
    Standarizes X and Y according to the given type
    
    Input
    -------
        - X, Y (DataFrame) : voxel-wise, behavior dataset 
        - type_ (str) : type of standarization (only z-score, z-score + normalization, z-score per films)
        - durations (array) : duration of each film used for the z-score per films standarization
    Output
    -------
        - X_normed, Y_normed (DataFrame) : standarized X & Y
    """
    X_normed = X.copy()
    Y_normed = Y.copy()
    
    if(type_=="zscore"): 

        X_normed =X_normed.apply(stats.zscore)
        Y_normed = Y_normed.apply(stats.zscore)
        
    elif(type_=="group_zscore"):
    
        
        if(durations != None):
            id_0=0
            id_1=durations[0]
            for idx in range(len(durations)-1):  
                X_normed[id_0:id_1]=X_normed[id_0:id_1].apply(stats.zscore)
                Y_normed[id_0:id_1]=Y_normed[id_0:id_1].apply(stats.zscore)
                print('youy')
                id_0=id_1
                id_1=id_1+durations[idx+1]

    return pd.DataFrame(X_normed), pd.DataFrame(Y_normed)

def R_cov(X, Y) : 
    """
    Computes the Correlation Matrix
    
    Input 
    -------
        - X (T x V Dataframe) : Voxel-wise serie
        - Y (T x M DataFrame) : Behavior dataset 
    Ouput
    -------
        - R (M x V Array) : Correlation Matrix
    """
    ## Check X & Y dimension
    if(X.shape[0] != Y.shape[0]): raise Exception("Input arguments X and Y should have the same number of rows")
        
    R = (np.array(Y.T) @ np.array(X))
    return R


def SVD(R, ICA=False,n_components=None, seed=4):
    """
    Singular Value Decomposition of R
    
    Input 
    -------
        - R (L (#LCs) x V Array) : Correlation Matrix
        - ICA (bool): if True turn LCs such that max is positive
        - n_component (int) : number of LCs to keep for the decomposition
        - seed (int)
    Output
    -------
        - U ( M X L (#LCs) DataFrame) : left singular vector matrix
        - S ( L x L (#LCs) DataFrame) : Diagonal Singular value matrix
        - V ( V x L (#LCs) Dataframe) :  Right Singular vector matrix Transposed
       
    """
   # R = np.array(R)
    n_components = min(R.shape)
    
    
    ## Run most computationally efficient SVD
    U, d, V = randomized_svd(R, n_components=n_components, random_state=seed)
    
    ## Get V instead of V.T
    V=V.T
    
    ## ICA convention
    result = np.where(np.abs(V)==np.amax(np.abs(V), axis=0))
    if(ICA):
        for i in range(len(result[0])):
            if(np.sign(V[result[0][i],result[1][i]]))<0 : 
                V[:,result[1][i]]=-V[:,result[1][i]]
                U[:,result[1][i]]=-U[:,result[1][i]]
    
    return pd.DataFrame(U), pd.DataFrame(np.diag(d)), pd.DataFrame(V)


def varexp(singular):
    """
    Computes the explained variance from the Singular values matrix 
   
    Input 
    -------
        - Singular (L x L (#LCs) DataFrame) : Singular matrix from SVD decomposition
    Ouptut
    -------
        - var (L(#LCs)x 1 vector) : explained variance for each singular value
    
    """
    ## Verify if matrix dimension 2x2 
    if singular.ndim != 2:
        raise ValueError('Provided DataFrame must be a square diagonal '
                         'matrix, not array of shape {}'
                         .format(singular.shape))
    ## Variance computation    
    var = (np.diag(singular)**2 / np.sum(np.diag(singular)**2))
    return var

def PLS_scores(X, Y, U, V):
    """
    Compute the PLS scores ("Brain" & "Design") by projecting the original data (X and Y) 
    onto their respective salience patterns (V and U)
    
    Input 
    -------
        - X (T x V DataFrame) : voxel-wise series 
        - Y (T x M DataFrame) : Emotional items 
        - U (M x L(#LCs)  DataFrame) : Left singular vector from SVD decomposition
        - V (V x L (#LCs)DataFrame) : Right Singular Vector from SVD decompositon (transposed)
    Output  
    -------
        - Lx (T x L(#LCs) Dataframe) : Imaging/Brain scores
        - Ly (T x L(#LCs) DataFrame) : Design/Behavior scores
    """
    Lx= X@np.array(V)
    Ly= Y@np.array(U)
    return Lx, Ly
        
def rotatemat (origlv, bootlv):
    """
    Compute Procrustean Transform (correction for axis rotation/reflection du)
    
    Input 
    -------
        - origlv : Original Matrix
        - bootlv : Matrix after resampling
    Output 
    -------
        - new_mat : New matrix with rotation correction to re-order Lvs as the original matrix
    
    """
    ## Define coordinate space between original and bootstrap LVs
    tmp=origlv.T@bootlv
    

    ## Orthogonalze space
    [V,W,U]=SVD(tmp);
   
    ## Determine procrustean transform
    new_mat=U@V.T

    return new_mat

def permu(X,Y,U,nPerms, seed=1):
    """
    Implementation of the Permutation testing for PLS
    
    Input 
    -------
        - X (T x V DataFrame): voxel-wise series (standarized)
        - Y (T x M DataFrame): Emotional items (standarized)
        - U (M x L(#LCs)  DataFrame) : Left singular vector from SVD decomposition
        - nPerms (int) :  number of permutation to perform
        - seed (int) 
    Output 
    -------
        - Sp_new (L (#LCs) x nPerms array): Permueted singular values, used to compute p-values to asses LCs significance
    """
    Sp_new=[]
    
    ## Check that dimensions of X & Y are correct
    if(X.shape[0] != Y.shape[0]): raise Exception("Input arguments X and Y should have the same number of rows")
    
    ## Fixe the seed
    rs = RandomState(MT19937(SeedSequence(seed)))
    
    ## Permutation
    for i in range(nPerms):
        
        Xp=X
        ## Permute Y rows 
        Yp=Y.sample(frac=1,replace=False, random_state=rs)
    
        ## Generate cross-covariance matrix between X and permuted Y
        Rp = R_cov(np.array(Xp), np.array(Yp))
 
        ## Singular value decomposition of Rp
        Up, Sp, Vp = SVD(Rp)

        ## Procrustes rotation
        rot_mat=rotatemat(U,Up)
    
        Up = Up @ Sp @ rot_mat
        
        Sp = (np.sqrt(np.sum(np.square(Up), axis = 0)))
       
        ## Concatenate the resulting Singular values matrix
        Sp_new.append(Sp)
        
    return np.array(Sp_new).T

def myPLS_get_LC_pvals(Sp_vect,S,nPerms, seuil=0.01) : 
    """
    Compute p-values for all Latent Components (LCs) using the permuted singular values
    
    Input  
    -------
        - Sp_new (L (#LCs) x nPerms array): Permueted singular values
        - S ( L x L (#LCs) DataFrame) : Diagonal Singular value matrix
        - nPerms (int) : Number of Permutation
        - seuil : significant level (0.01 per default)
    Output  
    -------
        - sprob (L(#LCs) x 1 vector) : 
        - sign_PLC (vector) : indexe(s) of significant LCs 
    """
    sig_PLC=[]
    
    ## Check the number of permutations with Sp_perm greater than S 
    sp = np.sum(Sp_vect > np.diag(S)[:, None], axis=1) + 1
    
    ## Compute p-value for each LC- approximation by counting 
    ## the number of permuted singular values above the measured singular values
    sprob = sp / (Sp_vect.shape[-1] + 1)
    
    ## Index of significant LCs
    signif_LC = np.argwhere(sprob<seuil)
    
    ## Number of significant LCs
    nSignifLC = signif_LC.shape[0]
    
    ## Display & Concatenate significant LCs
    for i in range(nSignifLC):
        sig_PLC.append(signif_LC[i])
        print(f"LC {sig_PLC[-1]} with p-value = {sprob[sig_PLC[-1]]} \n")
        
    return sprob, sig_PLC


        
def myPLS_bootstrapping(X0,Y0,U,V, nBoots,  seed=1, type_="zscore", durations=None):
    """
    Boostrap on X0 & Y0 and recompute SVD 
    
    Input 
    -------
    - X0 (T x V DataFrame) : Voxels-wise serie (not normalized)
    - Y0 (T x M DataFrame) : Behavior/design data (not normalized)
    - U (M x L(#LCs)  DataFrame) : Left singular vector from SVD decomposition
    - V (V x L (#LCs)DataFrame) : Right Singular Vector from SVD decompositon (transposed)
    - nBoots (int) : number of bootstrap sample 
    - seed (int)
    - type_ (str) : type of standarization (only z-scored, z-scored per films)
    - durations (array) : duration of each film used for the z-score per films standarization
    Output 
    -------
    - boot_results (dic) : containg results from Bootstrapping --> Ub_vect nboots x M x L matrix
                                                               --> Vb_vect nboots x V x L matrix
                                                               
    - boot_stat (dic) : containing statistique from Boostrapping --> bsr_u MxL matrix (DataFrame) storing stability score for U
                                                                 --> bsr_v VxL matrix (DataFrame) storing stability score for V
                                                                 --> u_std MxL matrix (Array) storing standard deviation for U
                                                                 --> v_std VxL matrix (Array) storing standard deviation for V 
    """
  
    rs = RandomState(MT19937(SeedSequence(seed)))
    boot_results = {}
    
    Ub_vect = np.zeros((nBoots,) + U.shape)
    Vb_vect= np.zeros((nBoots,) + V.shape)
    
    for i in range(nBoots):
        ## X & Y resampling
        Xb = X0.sample(frac=1, replace=True, random_state=rs)
        Yb = Y0.sample(frac=1, replace=True, random_state=rs)
        
        ## X & Y standardization
        # Xb, Yb = standarization(Xb, Yb, type_, durations)
       
        ## Cross-covariance
        Rb = R_cov(Xb, Yb)
        
        ## SVD 
        Ub, Sb, Vb = SVD(Rb)
        
        ## Procrustas transform (correction for axis rotation/reflection)
        rotatemat1 = rotatemat(U, Ub)
        rotatemat2 = rotatemat(V, Vb)
        
        ## Full rotation
        rotatemat_full = (rotatemat1 + rotatemat2) / 2
        Vb = Vb @ rotatemat_full
        Ub = Ub @ rotatemat_full
        
        ## Store Singular vectors
        Ub_vect[i] = Ub
        Vb_vect[i] = Vb
    
    boot_results['u_std'] = np.std(Ub_vect, axis=0)
    boot_results['v_std'] = np.std(Vb_vect, axis=0)
    
    boot_results['bsr_u'] = U / boot_results['u_std']
    boot_results['bsr_v'] = V / boot_results['v_std']

    return boot_results

        
            
def boot_select(LC_index, boot_res, X, level=2.5): 
    """
   
    Select the important voxels based on the boot stability scores
    
    Inputs
    -------
    LC_index : int
        integer indices of the latent variables
    boot_res : pandas DataFrame
        Dataframe with the bootstrap stability scores
    X : pandas DataFrame
        Dataframe with the original input data
    level : float
        The cutoff threshold for selecting voxels
    
    Ouput
    -------
    select_X : numpy array
        Array of selected voxels
    """
    select_X=np.zeros_like(X.iloc[:, LC_index])
    index = np.argwhere(np.array(abs(boot_res.iloc[:,LC_index]))>level)
    select_X[index[:,0]]=X.iloc[index[:,0], LC_index]
    return select_X