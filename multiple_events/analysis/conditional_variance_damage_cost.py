import numpy as np
import pandas as pd
import pygam as pg
import scipy.stats as stats
import scipy.interpolate as interp
import floodprediction as fp
import matplotlib.pyplot as plt
import pickle
import os

### *** HELPER FUNCTIONS *** ###

def money_label(x):
    
    if x >= 1e9:
        x = x/1e9
        label = f'${x:.1f}B'
    elif x >= 1e6:
        x= int(np.round(x/1e6))
        label = f'${x}M'
    else:
        x = int(np.round(x/1e3))
        label = f'${x}k'
        
    return(label)

### *** INITIAL SETUP *** ###

pwd = os.getcwd()

outfolder = os.path.join(pwd,'conditional_variance_models')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)
    
# Matplotlib default settings
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

### *** READ IN FLOODEVENT OBJECTS *** ###

print('*** FLOODEVENT OBJECTS ***',flush=True)

# Specify path to floodevent model runs
damage_dir = os.path.join(pwd,'2024-09-11_damage_estimates')

floodevent_filenames = [x for x in np.sort(os.listdir(damage_dir)) if '.object' in x]
floodevent_filepaths = [os.path.join(damage_dir,x) for x in floodevent_filenames]
floodevent_names = [x.split('_')[1] for x in floodevent_filenames]

floodevent_years = [pd.Timestamp(x.split('_')[0]).year for x in floodevent_filenames]
floodevent_labels = [f'{name}\n({year})' for name,year in zip(floodevent_names,floodevent_years)]
floodevent_list = []

for i,filepath in enumerate(floodevent_filepaths):
    
    print(floodevent_names[i],flush=True)
    
    with open(filepath, 'rb') as f:
        floodevent = pickle.load(f)
        
    floodevent_list.append(floodevent)
    
# *** FIT CONDITIONAL VARIANCE OF DAMAGE COST FOR EACH EVENT *** ###

conditional_variance_dict = {}

cv_x = []
cv_y = []

for floodevent in floodevent_list:

    df = floodevent.random_cv_predictions

    # Get properties that were flooded
    df = df[(df['flood_damage_class']==1)]
    x = df['cost_given_presence'].to_numpy()
    y = df['total_cost'].to_numpy()
    
    cv_x.append(x)
    cv_y.append(y)
    
    r = y-x
    r2 = r**2
    
    mod = pg.LinearGAM(pg.s(0),lam=10).fit(x,r2)

    xvals = np.arange(100*(np.min(x)//100),np.max(x),100)
    conditional_variance = mod.predict(xvals)
    
    # Set minimum bound on variance in damage estimate (e.g., +/- $5000)
    conditional_variance = np.maximum(conditional_variance,5000**2)
    
    f = interp.interp1d(xvals,conditional_variance,bounds_error=False,fill_value=(conditional_variance[0],conditional_variance[-1]))
    key = str(floodevent.start_date.to_period('M'))
    
    conditional_variance_dict[key] = f
    
# Save to file
outname = os.path.join(outfolder,'damage_cost_conditional_variance.pickle')
with open(outname, 'wb') as file:
    pickle.dump(conditional_variance_dict, file)
    
# *** CREATE PLOTS ILLUSTRATING RANGE OF UNCERTAINTY IN DAMAGE COST ESTIMATES *** ###

fig,axes = plt.subplots(nrows=4,ncols=2,figsize=(7.5,7.5))

ft = 9
alpha=0.1

labels=['Fran (1996)',
        'Bonnie (1998)',
        'Floyd (1999)',
        'Isabel (2003)',
        'Irene (2011)',
        'Matthew (2016)',
        'Florence (2018)']

meanvals = np.arange(100,250000,100)
LB = np.zeros(meanvals.shape)
UB = np.zeros(meanvals.shape)

for idx in range(7):
    
    label = labels[idx]
    
    ai = idx // 2
    aj = idx - 2*ai
    
    ax = axes[ai,aj]
    
    interp_func = list(conditional_variance_dict.values())[idx]
    
    for i in range(len(meanvals)):

        mean = meanvals[i]
        variance = interp_func(mean)

        mu = np.log(mean**2/np.sqrt(mean**2 + variance))
        sigma = np.sqrt(np.log(1+variance/mean**2))

        d = stats.lognorm(s=sigma,scale=np.exp(mu))

        LB[i] = d.ppf(alpha/2)
        UB[i] = d.ppf(1-alpha/2)
    
    ax.scatter(cv_x[idx],cv_y[idx],alpha=alpha,label='Cross-validation data')
    ax.plot(meanvals,LB,'r--',label='95% credible interval of fitted\nlognormal distribution')
    ax.plot(meanvals,UB,'r--')
    ax.plot(meanvals,meanvals,'k-',label='Mean of fitted lognormal\ndistribution')
    
    xmax = 50000*(1+(np.percentile(cv_x[idx],99) // 50000))
    xmax = min(xmax,250e3)
    
    if xmax > 100000:
        ax.set_xticks(np.arange(0,xmax+1,50000))
    
    ax.set_xlim([0,xmax])
    ax.set_ylim([0,None])
    
    xticklabels = [money_label(v) for v in ax.get_xticks()]
    yticklabels = [money_label(v) for v in ax.get_yticks()]
    
    ax.set_xticklabels(xticklabels,fontsize=ft)
    ax.set_yticklabels(yticklabels,fontsize=ft)
    
    ax.set_xlabel('Predicted cost, USD (nominal)',fontsize=ft)
    ax.set_ylabel('Observed cost,\nUSD (nominal)',fontsize=ft)
    ax.set_title(labels[idx],fontweight='bold',fontsize=ft+1)
    
# Plot one extra point that's off-screen to create a lengend entry for CV data that's less transparent
# (This is just a kluge to get the legend stuff formatted correctly--not actual data, and won't appear in plot)
ax.scatter([1e9],[1e9],color='C0',alpha=0.6,label='Cross-validation data')
lines,labels = ax.get_legend_handles_labels()
lines = list(np.flip(lines[1:]))
labels = list(np.flip(labels[1:]))

fig.legend(lines, labels, loc='lower right', bbox_to_anchor=(0.96,0.075))

axes[-1,-1].axis('off')

fig.tight_layout()

outname = os.path.join(outfolder,'damage_cost_conditional_variance.png')
fig.savefig(outname,dpi=400)

fig.show()