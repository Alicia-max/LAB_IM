from behavPLS import BehavPLS
from plot import*
from nilearn.masking import compute_brain_mask
from configparser import ConfigParser
from argparse import ArgumentParser


import pickle
import typer


app = typer.Typer()
import yaml

@app.command()

def load_pkl (data, name): 
    with open(f'../pkl/{name}.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open(f'../pkl/{name}.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        

def main(onset_dir,mask_dir,fmri_dir,behav_dir,films,nb,type_,nPer,nBoot,seed):


   
   
    dataset=BehavPLS(onset_dir,mask_dir,fmri_dir, behav_dir , films, nb, type_, nPer, nBoot)
   
 
    res_decompo = dataset.run_decomposition()
    load_pkl(res_decompo, f"pls_res_{type_}")
   
   
    res_permu=dataset.permutation()
    load_pkl(res_permu, f"perm_res_{type_}")
        
    res_bootstrap = dataset.bootstrap()
    load_pkl(res_bootstrap, f"boot_res_{type_}")

if __name__ == "__main__":
   
    
    parser =  ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/Appraisal.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)
        onset_dir = config["onset_dir"]
        mask_dir = config["mask_dir"]
        fmri_dir = config["fmri_dir"]
        behav_dir = config["behav_dir"]
        films = config["films"]
        nb = config["nb"]
        type_ = config["type_"]
        nPer = config["nPer"]
        nBoot = config["nBoot"]
        seed = config["seed"]
        
    
    
    main(onset_dir,mask_dir,fmri_dir,behav_dir,films,nb,type_,nPer,nBoot,seed)
    
    
    

    
