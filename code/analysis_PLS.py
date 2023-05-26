from behavPLS import BehavPLS
from plot import*
from nilearn.masking import compute_brain_mask
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
        

    
def main(config_file):
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
        onset_dir = config["onset_dir"]
        mask_dir = config["mask_dir"]
        fmri_dir = config["fmri_dir"]
        behav_dir = config["behav_dir"]
        films = config["films"]
        nb = config["nb"]
        type_ = config["type_"]
        nPer = config["nPer"]
        nBoot = config["nBoot"]
        norma = config["norma"]
        seed = config["seed"]
   
    dataset=BehavPLS(onset_dir,mask_dir,fmri_dir, behav_dir , films, nb, type_, nPer, nBoot, norma)
    res_decompo = dataset.run_decomposition()
    load_pkl(res_decompo, f"pls_res_{type_}_{norma}")
    
    res_permu=dataset.permutation()
    load_pkl(res_permu, f"perm_res_{type_}_{norma}")
        
    res_bootstrap = dataset.bootstrap()
    load_pkl(res_bootstrap, f"boot_res_{type_}_{norma}")
    
    print("...Behav Plot...")
    plot_behav(res_permu['sig_LC'] , res_decompo['Y'].columns, res_decompo['U'],  f"{type_}_{norma}", res_bootstrap['std_u'])
        
    print("... Brain Plot..")    
    mask= compute_brain_mask(nib.load(os.path.join("../reg", 'gray_matter.nii.gz')))
    for LC in res_permu['sig_LC']:
            V_final = boot_select(LC[0],res_bootstrap["bsr_v"], res_decompo['V'] )
            save_fMRI(V_final,mask , f"LV{LC[0]+1}_{type_}_{norma}")
            
    brain_plot(res_permu['sig_LC'], f"{type_}_{norma}")
       
 
    
    return res_decompo,res_permu, res_bootstrap
   

if __name__ == "__main__":
    app()
    
    
