from compute import*
from plot import*


class BehavPLS(): 
    """
    Parameters : 
    ----
     Brain Data - X () : 
     Behav Data - Y() : 
     Behav Names
     Image name :
     res
     
    
    """
    
    def __init__(self, directory, onset_TR=72, verbose=True, mean=False,nPerms= 100, nBoot=100, seed=1, seuil=0.01,  **kwargs):
        
        if(verbose) : print("... Initilization Behavior Dataset ...\n")
        
        
        
        # Behavior Data 
        dir_behav_lab = os.path.join(directory,"Annot_Sintel_stim.json")
        self.behav_labels =  json.loads(open(dir_behav_lab).read())
        
        dir_behav_annot = os.path.join(directory,"Annot_Sintel_stim.tsv")
        self.behav_data=pd.read_csv(dir_behav_annot, sep='\t', names =self.behav_labels['Columns'])
        
        
        if(verbose) : print("... Resampling Behavior Dataset...\n")
        
        #Resampling 
        self.behav_data = resampling(self.behav_data)
        
        # Brain Data 
        if(verbose) : print("... Initilization Brain Dataset ...\n")
        dir_sub = os.path.join(directory,"TC_sub14_labels.csv")
        with open(dir_sub) as file : 
            name_sub_cortical=file.read().splitlines() 
        dir_cor = os.path.join(directory,"TC_cort400_labels.csv")
        with open(dir_cor) as file: 
            name_corticale=file.read().splitlines() 
            
        self.brain_labels=[name_corticale+name_sub_cortical]
        
        brains_csv=[]    
        for brain_file in sorted(os.listdir(directory)):
            if brain_file.endswith('Sintel.csv'): brains_csv.append(brain_file)  
            
        #Sort file per subject    
        brains_csv.sort(key=lambda x:  int(x.rsplit('_',4)[2].rsplit('S')[1]))
        
        if(verbose) : print("...Onset & Offset Removal for Brain dataset...\n")
        #Onset & Offset removing
        offset_TR=len( self.behav_data)
        df_= pd.concat([alignement(pd.read_csv(directory+f, sep=',', decimal='.'), offset_TR, onset_TR) for f in brains_csv], axis=1, ignore_index=True)
        df_.columns=self.brain_labels[0]*6
        
        #average voxels activation across subject 
        if(mean) : 
            if(verbose) : print("...Average across subject...\n")
            
            mean=pd.concat([df_[i].mean(axis=1) for i in name_corticale+name_sub_cortical], axis = 1)
            mean.columns=self.brain_labels
            self.brain_data =mean
        else : self.brain_data = df_
        
        self.nPerms = nPerms
        self.nBoot=nBoot
        self.seed=seed
        self.seuil=seuil
        
    
        
    def run_pls(self,**kwargs): 
        res={}
        
        print("norma")
        self.brain_std, self.behav_std = standarization (self.brain_data, self.behav_data)
        
        print("...SVD ...")
        res['R']=R_cov(self.brain_std, self.behav_std)
        res['U'],res['S'], res['V'] = SVD(res['R'], ICA=True)
        res['ExplainedVarLC']=varexp(res['S'])
        res['Lx'], res['Ly']= PLS_scores(self.brain_std, self.behav_std, res['U'], res['V'])
        print("...Permu...")
        res['Sp_vect']=permu(self.brain_std, self.behav_std, res['U'], self.nPerms, self.seed)
        res['P_val'], res['sig_LC'] = myPLS_get_LC_pvals(res['Sp_vect'],res['S'],self.nPerms, self.seuil)
        
        exp_var(res['S'], res['Sp_vect'],  res['P_val'], nc =10)
        brain_plot(np.arange(6), res['V'])
        spiralplot(res['U'], self.behav_data,np.arange(6))
        
        return res
 
            
            
        
    
        
        
