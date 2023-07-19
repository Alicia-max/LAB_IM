from compute import*
from plot import*
import pickle
import yaml

class Emotions_PLS() : 
    
    def __init__(self, discrete_pkl, appraisal_pkl,  nPerms= 100, nBoot=100, seed=1, seuil=0.01): 
        
        
        discrete = pd.read_pickle(discrete_pkl)
        appraisal = pd.read_pickle(appraisal_pkl)
        
        
        self.X=discrete['Y']
        
        self.Y = appraisal['Y']
        self.X_std = discrete['Y_std']
        self.Y_std = appraisal['Y_std']
        self.dur = discrete['sub_time']
        self.nPerms = nPerms
        self.nBoot=nBoot
        self.seed=seed
        self.seuil=seuil
        
    def run(self): 
        res={}
        
        print("... SVD...")
        self.R=R_cov(self.X_std, self.Y_std)
        self.U,self.S, self.V = SVD(self.R, ICA=True)
        self.ExplainedVarLC =varexp(self.S)
        res['R']=self.R
        res['U']=self.U
        res['S']=self.S
        res['V']=self.V
        
        print("...Permu...")
        res['Sp_vect']=permu(self.X_std, self.Y_std, self.U, self.nPerms, self.seed)
        res['P_val'], res['sig_LC'] = myPLS_get_LC_pvals(res['Sp_vect'],self.S,self.nPerms, self.seuil)
        
        print("... Bootstrap...")
        res['boot'] = myPLS_bootstrapping(self.X,self.Y,self.U,self.V, self.nBoot, self.dur, seed = self.seed)
        return res
    
def main(config_file):
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
        discrete_pkl = config["discrete_pkl"]
        appraisal_pkl = config["appraisal_pkl"]
        nPerms = config["nPerms"]
        nBoot = config["nBoot"]
        seed = config["seed"]
        seuil = config["seuil"]
        
        
    
   
    dataset=Emotions_PLS(discrete_pkl, appraisal_pkl,nPerms, nBoot, seed, seuil)
    
    res=dataset.run()
    
    with open(f'../pkl/Discrete_Appraisal_PLS.pkl', 'wb') as f:
        pickle.dump(res, f)
    with open(f'../pkl/Discrete_Appraisal_PLS.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        

    plot_behav(res['sig_LC'], dataset.Y.columns ,res['U']*-1, "Appraisal_3rd_PLS", res['boot']['bsr_u'],
           np.transpose(res['boot']['u_std']), 'lightsteelblue',  'steelblue')
    
    plot_behav(res['sig_LC'], dataset.X.columns ,res['V']*-1, "Discrete_3rd_PLS", res['boot']['bsr_v'],
           np.transpose(res['boot']['v_std']), 'yellowgreen', "seagreen")
          
