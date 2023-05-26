from compute import*
import nibabel as nib
import os
import nilearn
import nibabel as nib
from nilearn import plotting, image
import numpy as np
import matplotlib.pyplot as plt

def exp_var(S, Sp_vect, LC_pvals, name): 
    """
    Plot the cumulative explained variance and save it
    
    Inputs 
    -------
    S: ndarray
        Array of singular values
        
    Sp_vect: ndarray
        Array of LC vectors
        
    LC_pvals: ndarray
        Array of LC p-values
        
    name: str
        Name of the plot file to save
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Number of LCs
    nc = np.arange(len(LC_pvals)) + 1
    
    # Create another axes object for secondary y-Axis
    ax2 = ax.twinx()
    
    # Plot singular values
    ax.plot(nc, np.diag(S), color='grey', marker='o', fillstyle='none')
    
    # Plot the cumulative explained covariance
    ax2.plot(nc, (np.cumsum(varexp(S))*100), color='seagreen', ls='--')
    
    # Labeling & Axes limits
    labels = [f"LC {idx+1} (p={np.round(i, 4)})" for idx, i in enumerate(LC_pvals)]
    plt.title('Explained Covariance by each LC')
    ax.set_ylabel('Singular values')
    ax.set_xticklabels(labels, rotation=80)
    ax.set_xlim([1, len(LC_pvals)])
    ax2.set_xticks(nc, labels)
    ax2.set_ylabel('Explained correlation', color='seagreen')
    ax2.set_ylim([0, 100])

    # Defining display layout
    plt.tight_layout()
    
    # Save the Plot
    plt.savefig(f"../Plots/Var/{name}.png", bbox_inches='tight')

def save_fMRI(data, mask, title, shape=[91, 109, 91]):
    """
    Convert and save the received data into a Nifti file
    
    Inputs:
    -------
    data: np.array
            input data
    mask: nib.Nifti1Image
            the binary mask
    title: str
            the title of the saved file
    shape: list
            shape of the Nifti file (default=[91, 109, 91])
    """
    # Create an array of the size of the mask
    new_brain = np.zeros(np.shape(mask.get_fdata() > 0))
    
    # Prune voxels to 0 if outside the mask
    new_brain[mask.get_fdata() > 0] = data
    new_brain = new_brain.reshape(shape)
    
    # Convert to Nifti format and save
    nift_fMRI = nib.Nifti1Image(new_brain, mask.affine)
    nib.save(nift_fMRI, f"../Nifti/{title}.nii.gz")
    
def plot_z_slices(img, n_rows, n_cols, title=None, output_file=None):
    """
    Plot axial slices of a 3D NIfTI image.
    
    Inputs:
    -------
    img : nibabel.nifti1.Nifti1Image
        Input 3D NIfTI image.
    n_rows : int
        Number of rows for the plot.
    n_cols : int
        Number of columns for the plot.
    title : str, optional
        Title of the plot.
    output_file : str, optional
        Name of the output file to save the plot.
    """
    all_coords = plotting.find_cut_slices(img, direction="z", n_cuts=n_rows * n_cols)
    ax_size = 3.0
    margin = 0.05
    fig, all_axes = plt.subplots(n_rows, n_cols, figsize=(n_rows * ax_size, n_cols * ax_size),
                                 gridspec_kw={"hspace": margin, "wspace": margin})
    left_right = True
    for coord, ax in zip(all_coords, all_axes.ravel()):
        display = plotting.plot_stat_map(img, cut_coords=[coord], display_mode="z", axes=ax, annotate=False)
        display.annotate(left_right=left_right)
        left_right = False
    if title:
        plt.suptitle(title, fontsize=15)
    if output_file:
        plt.savefig(output_file)
        
def brain_plot_slices(LC_indexes, name): 
    """
    Plot and save brain saliences 
    """
    for i in LC_indexes: 
        file = f"../Nifti/LV{i+1}_{name}.nii.gz"
        pic = image.load_img(file)
        plot_z_slices(pic, 5, 5, f"Brain Z-slices, LV: {i+1}", f"../Plots/Brain/Brain Z-slices,LV:{i+1}_{name}")

        
def brain_plot(LC_indexes, name): 
    """
    Plot and Save Brain Saliences 
    
    Inputs
    -------
    LC_indexes : list
        List of tuples with the indexes of the LVs to plot
    name : str
        Name of the saliency map
    
    """
    for i in LC_indexes:
        file = f'../Nifti/LV{i[0]+1}_{name}.nii.gz'
        pic = image.load_img(file)
        plotting.plot_stat_map(pic,
                               cmap = 'coolwarm',
                               display_mode='ortho',
                               draw_cross=False, 
                               cut_coords=(37,-60,8),
                               annotate=True,
                               black_bg=False)
        plt.savefig(f'../Plots/Brain/LV{i[0]+1}_{name}.png')
        
        
def plot_behav(LC_indexes, x, U, name, boot_std): 
    """
    Plot and Save the Behavioral Saliences
    
    Inputs
    -------
    LC_indexes : list
        List of tuples with the indexes of the LVs to plot
    x : array-like
        Array of labels for the behavioral variables
    U : array-like
        Matrix of behavioral saliencies
    name : str
        Name of the saliency map
    """
    for i in LC_indexes :
        f, ax = plt.subplots(figsize=(6,3))
        fig=plt.bar(x, U[i[0]], yerr=boot_std[i[0]], color='lightseagreen')
        ax.set_ylabel(f"Loadings {i[0]+1}")
        plt.xticks(rotation=80)
        plt.savefig(f"../Plots/Behav/Behav{i[0]+1}_{name}.png", bbox_inches='tight')



