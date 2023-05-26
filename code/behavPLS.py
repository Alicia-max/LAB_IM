from compute import*
import nibabel as nib
import glob
from nilearn.image import clean_img
from nilearn.masking import compute_brain_mask, apply_mask


class BehavPLS(): 
    '''
    Parameters : 
    ----
    '''
    def __init__(self, onset_dir, mask_dir, fmri_dir, behav_dir, films, nb_sub, type_,
                 nPerms= 100, nBoot=100, norma_type="zscore", seed=1, seuil=0.01,verbose=True,  **kwargs): 
        
        
        self.brain_data, self.durations = self.get_brain_data(onset_dir, mask_dir, fmri_dir, films, nb_sub, verbose)
        
        self.behav_data = self.get_behav_data(behav_dir, self.durations, films, verbose)
        
        if(type_=="Appraisal"):
            self.behav_data = self.behav_data.drop(self.behav_data.columns[10:,], axis=1)
        if(type_=="Discrete") : 
            self.behav_data = self.behav_data.drop(self.behav_data.columns[0:37], axis=1)
        if (type_=="Expression") :
            self.behav_data=self.behav_data[self.behav_data.columns[10:15]]
        if(type_ == "Motivatoin") :
            self.behav_data=self.behav_data[self.behav_data.columns[15:25]]
        if(type_=="Feelings") :
            self.behav_data=self.behav_data[self.behav_data.columns[25:32]]
        if(type_=="Physiology"):
            self.behav_data=self.behav_data[self.behav_data.columns[32:37]]
        
        elif(type_=="all"): 
            self.behav_data=self.behav_data
            
            #all without discret 1:27
            
        self.norm = norma_type
        self.nPerms = nPerms
        self.nBoot=nBoot
        self.seed=seed
        self.seuil=seuil
        self.type=type_

        
    def run_decomposition(self, **kwargs): 
        res={}
        print("... Normalisation ...")
        self.X_std, self.Y_std = standarization(self.brain_data, self.behav_data, self.norm, self.durations)
        res['durations']=self.durations
        res['X']=self.brain_data
        res['Y']=self.behav_data 
        res['X_std']= self.X_std
        res['Y_std']= self.Y_std
     
        print("...SVD ...")
        self.R=R_cov(self.X_std, self.Y_std)
        self.U,self.S, self.V = SVD(self.R, ICA=True)
        self.ExplainedVarLC =varexp(self.S)
        self.Lx, self.Ly= PLS_scores(self.X_std, self.Y_std, self.U, self.V)
        
        
        res['R']=self.R
        res['U']=self.U
        res['S']=self.S
        res['V']=self.V
        res['Lx']=self.Lx
        res['Ly']=self.Ly
        return res
        

    def permutation(self, **kwargs):
        print("...Permu...")
        res={}
        res['Sp_vect']=permu(self.X_std, self.Y_std, self.U, self.nPerms, self.seed)
        res['P_val'], res['sig_LC'] = myPLS_get_LC_pvals(res['Sp_vect'],self.S,self.nPerms, self.seuil)
        
        return res 
    
    def bootstrap(self, **kwargs): 
        print("... Bootstrap...")
        res={}
        res['boot'] = myPLS_bootstrapping(self.brain_data,self.behav_data , 
                                                                self.U,self.V, self.nBoot,  self.seed, 
                                                                 self.norm, self.durations)
       
        return res
 
    def get_brain_data(self, onset_dir, mask_dir, fmri_dir, films, nb_sub, delay=4, verbose=False): 
        
      
        sub_ID=['%0*d' %(2, i+1) for i in np.arange(nb_sub)]
        mean_vox=[]
        durations = []
        
        
        sub_ID.remove('12')
        sub_ID.remove('18')
        
        ## Get Mask 
        mask = compute_brain_mask(nib.load(os.path.join(mask_dir, 'gray_matter.nii.gz')))
        
        for f_idx,f in enumerate(films):
            vox_film=[]
            for ID in sub_ID: 
                
                
                ## Get Files name
                try : 
                    o_f = os.path.join(onset_dir,f"sub-S{ID}/**/*{f}_events.tsv*")
                    o_f=glob.glob(o_f, recursive=True)
                    p_s = os.path.join(fmri_dir,f"sub-S{ID}/**/*{f}.feat*")
                    p_s=glob.glob(p_s, recursive=True)
                except:
                        print("Something went wrong with file reading ")
                        
                ## Read Onset File
                o_f=pd.read_csv(o_f[0], sep='\t')
                onset=int(np.round(o_f[o_f['trial_type']=="film"]['onset'])+delay)
                duration = int(np.round(o_f[o_f['trial_type']=="film"]['duration']/1.3))
            
                ## Read fMRI File
                for file in sorted(os.listdir(p_s[0])):
                    if file.endswith('MNI.nii'):
                        
                        ## Apply Mask
                        map_ = nib.load(os.path.join(p_s[0], file))
                        x=apply_mask(clean_img(map_, standardize=False, ensure_finite=True), mask)
                        
                        if (ID == "01" and f_idx == 12) :
                            duration = 309
                        if( ID == "31" and f_idx == 3) : 
                            duration = 312
                        ## Onset Removed
                        x = x[onset:onset+duration-1]

                        ## Scrubbing
                        vox_film.append(self.scrubbing(p_s[0], onset, duration, x, True, 0.5))
                        
          
            ## Average among subject
           
            if(verbose) : print("... Brain Data Loading ...")
            mean_vox.append(np.nanmean(vox_film, axis=0))
            durations.append(duration)    
            
        mean_vox=np.vstack(mean_vox)    
        X=pd.DataFrame(np.array(mean_vox).reshape(-1, np.array(mean_vox).shape[-1]))
        return X, durations

    def get_behav_data(self,directory, durations, films, verbose):
        
        '''
        Input : 
        --------
            -directory : 
            -durations : 
            - films : 
            
        Output : 
        --------
            -behavs : 
            
        '''
        dir_behav_annot=[]
        for film in films: 
            dir_behav_lab = os.path.join(directory,f"Annot_{film}_stim.json")
            labels= json.loads(open(dir_behav_lab).read())
            dir_behav_annot.append(os.path.join(directory,f"Annot_{film}_stim.tsv"))
            
        ## Verify alphabetic order
        #dir_behav_annot.sort(key=lambda x:  str(x.rsplit('_',3)[1])) 
        
        if(verbose) : print("... Behavior Data Loading ...")
        ## Modify resampling method to cut at duration 
        behavs=pd.concat([resampling(pd.read_csv(f, sep='\t', names =labels['Columns']),dur)
                              for f, dur in zip(dir_behav_annot, durations)])
        
        return behavs
  

    def scrubbing(self, folder, onset, dur, vox, verbose=False, level=0.1):
        ## Read Motion File
        mc = pd.read_csv(f"{folder}/mc/prefiltered_func_data_mcf_rel.rms", header=None)
    
        ## Get index to remove & keep 
        mc=mc.iloc[onset:dur+onset-1].reset_index(drop=True)
        rmove=np.where(mc>level)[0]
        keep= np.where(mc<level)[0]
    
        ## Replace wrong values with None 
        if(len(rmove!=0)):
            pcs=len(keep)/(len(rmove)+len(keep))
            vox[rmove]=np.nan
            if(verbose) : print(f"% of data keep: {pcs*100} \n")
            
        return vox
 