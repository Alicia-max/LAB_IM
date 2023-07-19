from compute import*
import nibabel as nib
import glob
import os
from nilearn.image import clean_img
from nilearn.masking import compute_brain_mask, apply_mask
from collections import defaultdict


class BehavPLS(): 
    '''
    Parameters : 
        - onset_dir : Path corresponding to the directory for the onset files
        - mask_dir : Path corresponding to the directory for the mask file
        - fmri_dir : Path corresponding to the directory for the fMRI files
        - behav_dir : Path corresponding to the directory for the Behavioral Annotations files
        - films : list of str with the name of films used for the analysis
        - nb_sub : int, number of participant to take into account
        - type_ : str, specifcy the type of behavorial data to used (Appraisal only, Discrete only, Expression ect..)
        - nPerms : int, number of permutation for the permutation testing
        - nBoot : int, number of permutation for the bootstraping testing
        - seed : int, specify the seed for the resampling of the dataset (Permutation & Bootstraping)
        - seuil : int, signficant level for statistical testing
    ----
    '''
    
    
    def __init__(self, onset_dir, mask_dir, fmri_dir, behav_dir, films, nb_sub, type_,
                 nPerms= 100, nBoot=100, seed=1, seuil=0.01,verbose=True,  **kwargs): 
        
        
        self.brain_data, self.film_dur, self.dur_subj = self.get_brain_data(onset_dir, mask_dir, 
                                                                           fmri_dir, films, nb_sub, verbose)
       
                                                                         
        
        self.behav_data = self.get_behav_data(behav_dir, self.film_dur, films, verbose)
        
        if(type_=="test"): 
            self.behav_data=self.behav_data[self.behav_data.columns[15:25]]
        

        if(type_=="Appraisal"):
            self.behav_data = self.behav_data.drop(self.behav_data.columns[10:,], axis=1)
        if(type_=="Discrete") : 
            self.behav_data = self.behav_data.drop(self.behav_data.columns[0:37], axis=1)
        if (type_=="Expression") :
            self.behav_data=self.behav_data[self.behav_data.columns[10:15]]
        if(type_ == "Motivation") :
            self.behav_data=self.behav_data[self.behav_data.columns[15:25]]
        if(type_=="Feelings") :
            self.behav_data=self.behav_data[self.behav_data.columns[25:32]]
        if(type_=="Physiology"):
            self.behav_data=self.behav_data[self.behav_data.columns[32:37]]
        if (type_=="CPM_compo") : 
            self.behav_data=self.behav_data[self.behav_data.columns[0:37]]
        
        elif(type_=="all"): 
            self.behav_data=self.behav_data

        self.nPerms = nPerms
        self.nsub=nb_sub
        self.nBoot=nBoot
        self.seed=seed
        self.seuil=seuil
        self.type=type_

        
    def run_decomposition(self, **kwargs): 
        """
        
        """
                                            
                                            
        res={}
        
        print("... Normalisation ...")
        self.X_std, self.Y_std = standa(self.brain_data, self.behav_data, self.dur_subj)
        res['X']=self.brain_data
        res['Y']=self.behav_data 
        res['time']=self.film_dur
        res['sub_time']=self.dur_subj
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
        res= myPLS_bootstrapping(self.brain_data,self.behav_data , self.U,self.V, 
                                          self.nBoot,  self.film_dur, self.seed)
       
        return res
    
    def get_onset_file(self,onset_dir, fmri_dir, films, nb_sub):
        sub_ID=['%0*d' %(2, i+1) for i in np.arange(nb_sub)]
        files=defaultdict(list)
        
        
        if '18' in  sub_ID : 
            sub_ID.remove('18')
            nb_sub -=1
            
        if '12' in sub_ID :  
            sub_ID.remove('12')
            nb_sub-=1
            
        for f in (films):
            for ID in (sub_ID): 
                o_f_ = os.path.join(onset_dir,f"sub-S{ID}/**/*{f}_events.tsv*")
                files[f].append(glob.glob(o_f_, recursive=True)[0])  
                fMRI_ = os.path.join(fmri_dir,f"sub-S{ID}/**/*{f}.feat*") 
                files[f].append(glob.glob(fMRI_, recursive=True)[0])
        return files, nb_sub
 
    def get_brain_data(self, onset_dir, mask_dir, fmri_dir, films, nb_sub, delay=4, verbose=False): 
        
        dico_files, nb_sub = self.get_onset_file(onset_dir,  fmri_dir, films, nb_sub)
        mask = compute_brain_mask(nib.load(os.path.join(mask_dir, 'gray_matter.nii.gz')))
        
        
        mean_vox=[]
        durations_film=[]
        durations_sub_mean=[]
    
        for film, files in dico_files.items(): 
            vox_film=[]
            durations_sub=[]
            
            for name in files : 
                
                if(name.endswith('.tsv')) : 
                    o_f=pd.read_csv(name, sep='\t')
                    onset=int(np.round(o_f[o_f['trial_type']=="film"]['onset'])+delay)
                    ID = name.split("/", 5)[3].split('S')[1]
                    duration = int(np.round(o_f[o_f['trial_type']=="film"]['duration']/1.3))
                    
                if(name.endswith(".feat")):
                    for file in sorted(os.listdir(name)):
                        if file.endswith('MNI.nii'):
                            map_ = nib.load(os.path.join(name, file))
                            x=apply_mask(clean_img(map_, standardize=False, ensure_finite=True), mask)
                            
                            
                            if (ID == "01" and film == "ToClaireFromSonny") :duration = 309
                            if( ID == "31" and film == "Chatter") : duration = 312
                                
                            x = x[onset:onset+duration-1]    
                            new_x = scrubbing(name, onset, duration, x, True, 0.5)
                            durations_sub.append(len(new_x))
                            print(f"Time course for  the film {film} - subject{ID} :{len(new_x)}")
                            vox_film.append(new_x)
            
            vox_arr =np.vstack(vox_film)
            zscore_df = (vox_arr -np.nanmean( vox_arr,axis=0)) / np.nanstd( vox_arr,axis=0, ddof=0)
            split_z = np.vsplit(zscore_df,nb_sub)
            mean_vox.append(np.nanmean(split_z, axis=0, dtype="float32"))
            durations_film.append(duration)
            durations_sub_mean.append(int(np.mean(durations_sub)))
            
        
        mean_vox=np.vstack(mean_vox)     
        X=pd.DataFrame(np.array(mean_vox))
        return X, durations_film, durations_sub_mean
 

    def get_behav_data(self,directory, durations, films, verbose=False):
        
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
        
        if(verbose) : print("... Behavior Data Loading ...")
        ## Modify resampling method to cut at duration 
        behavs=pd.concat([resampling(pd.read_csv(f, sep='\t', names =labels['Columns']),dur)
                              for f, dur in zip(dir_behav_annot, durations)])
        
        return behavs
     
  

