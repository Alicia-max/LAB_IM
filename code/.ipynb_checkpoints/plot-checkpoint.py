from compute import*
import matplotlib.pyplot as plt
import plotly.express as px
from utils_neuromaps_brain import *



def exp_var(S, Sp_vect, LC_pvals, nc =10 ): 
    fig, ax = plt.subplots(figsize = (10, 5))
    plt.title('Explained Covariance by each LC')
    nc=nc
    
    # using the twinx() for creating another
    # axes object for secondary y-Axis
    ax2 = ax.twinx()
    ax.plot(np.arange(nc)+1, np.diag(S)[0:nc], color = 'grey', marker='o', fillstyle='none')
    ax.errorbar(np.arange(nc)+1, np.mean(Sp_vect, axis=1)[0:nc], np.std(Sp_vect, axis=1)[0:nc])
    ax2.plot(np.arange(nc)+1, (np.cumsum(varexp(S))*100)[0:nc], color = 'seagreen', ls='--')
    
    # giving labels to the axises
    ax.set_ylabel('Singular values')
    ax.set_xlim([1,nc])
    ax.set_xlabel('Latent components')
    
    # secondary y-axis label
    ax2.set_ylabel('Explained covariance', color = 'seagreen')
    ax2.set_ylim([0,100])
    
    # defining display layout
    plt.tight_layout()
    
    # show plot
    plt.show()
    plt.savefig("../images/Explained_cov.png")
    print(f"p-val : {LC_pvals[0:nc]}")
    
def brain_plot(LC_index, V): 
    mean=[]
    for i in range(414):
        indexes=np.arange(0+i,414*6+i, 414)
        mean.append(V.iloc[indexes].mean(axis=0))
        
    for i in LC_index : 
        fig=normal_view(pd.DataFrame(mean).iloc[:,i],edges=True,exp_form=False,q_thresh=0,parcellation=400,
                    xlabel='',brightness=0.7,graymap_rev=False,alpha_graymap=0.5)
        fig.savefig(f"../images/Brain Image LC {i}")
        
def spiralplot(U, Y, LC_index ):
    for i in LC_index :
        x = U.iloc[:,i]
        y = (Y.columns.values)
        df = pd.DataFrame(np.column_stack((x,y)),columns=['LVs','Emotions'])
        
        fig_LC = px.line_polar(df,   r="LVs", theta="Emotions",  
                               color_discrete_sequence=px.colors.sequential.Pinkyl_r,
                               width=800, height=800)
        fig_LC.show()
   
 
   