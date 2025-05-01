import os
import gc
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime as dt
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.stats as stats
import sklearn.metrics as metrics
import spacetimekriging as stk
import floodprediction as fp
import dependence_modeling as dm
import mortgage_model as mm
import itertools


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


### *** SPECIFY COUNTIES INCLUDED IN STUDY AREA *** ###

study_area_counties = ['Alamance','Alexander','Alleghany','Anson','Ashe','Beaufort','Bertie','Bladen','Brunswick','Cabarrus','Caldwell','Camden','Carteret','Caswell','Chatham','Chowan','Columbus','Craven','Cumberland','Currituck','Dare','Davidson','Davie','Duplin','Durham','Edgecombe','Forsyth','Franklin','Gates','Granville','Greene','Guilford','Halifax','Harnett','Hertford','Hoke','Hyde','Iredell','Johnston','Jones','Lee','Lenoir','Martin','Mecklenburg','Montgomery','Moore','Nash','New Hanover','Northampton','Onslow','Orange','Pamlico','Pasquotank','Pender','Perquimans','Person','Pitt','Randolph','Richmond','Robeson','Rockingham','Rowan','Sampson','Scotland','Stanly','Stokes','Surry','Tyrrell','Union','Vance','Wake','Warren','Washington','Watauga','Wayne','Wilkes','Wilson','Yadkin']


### *** SET UP FOLDERS AND ENVIRONMENT *** ###

# Specify current working directory
pwd = os.getcwd()

# Specify output directory for figures and tables
outfolder = os.path.join(pwd,dt.datetime.today().strftime('%Y-%m-%d_figures_and_tables'))
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

### *** GEOSPATIAL DATA SOURCES *** ###

# Specify CRS for maps
crs = 'EPSG:32617'

# Read in data on NC counties
counties_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/NC_counties'
counties = gpd.read_file(counties_path).to_crs(crs)
counties = counties[['FIPS','County','geometry']].rename(columns={'FIPS':'countyCode','County':'countyName'})

# Read in data on 2010 NC census tracts
census_tracts_path = f'/proj/characklab/flooddata/NC/multiple_events/geospatial_data/TIGER/nc_2010_census_tracts_clean'
census_tracts = gpd.read_file(census_tracts_path).to_crs(crs)
census_tracts = census_tracts.rename(columns={'GEOID':'censusTract_2010'})

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


### *** FIGURE S1 *** ###

fig,axes = plt.subplots(nrows=4,ncols=2,figsize=(7.5,7.5))

labels=['Hurricane Fran:\nSept 2 – 17, 1996',
        'Hurricane Bonnie:\nAug 22 – 31, 1998',
        'Hurricane Floyd:\nSept 8 – 26, 1999',
        'Hurricane Isabel:\nSept 16 – 24, 2003',
        'Hurricane Irene:\nAug 24 – Sept 10, 2011',
        'Hurricane Matthew:\nOct 5 – Oct 17, 2016',
        'Hurricane Florence:\nSept 8 – Sept 27, 2018']

for i in range(7):
    
    label = labels[i]
    
    ai = i // 2
    aj = i - 2*ai
    
    ax = axes[ai,aj]
    
    study_area = floodevent_list[i].study_area
    included_counties = counties[counties['geometry'].centroid.intersects(study_area)]

    counties.plot(ax=ax,facecolor='none',edgecolor='k',alpha=1,lw=1.2)
    included_counties.plot(ax=ax,facecolor='grey',edgecolor='k',alpha=0.7,lw=1.2)
    
    ax.set_title(label,fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
axes[-1,-1].axis('off')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S1_event_boundaries.png')
fig.savefig(outname,dpi=400)

fig.show()


# Helper functions to calcualate cross-validation performance of Random-Forest model

def get_performance(predictions_df):
    
    cost_response_variable = 'total_cost'
    presence_response_variable = 'flood_damage'

    y_pred = predictions_df[f'{presence_response_variable}_prob'].to_numpy()
    y_class = predictions_df[f'{presence_response_variable}_class'].to_numpy()
    y_true = predictions_df[presence_response_variable].to_numpy()
    c_pred = predictions_df[f'{cost_response_variable}_pred'].to_numpy()
    c_true = predictions_df[cost_response_variable].to_numpy()

    results_dict,roc_curve,pr_curve = fp.performance_metrics(y_pred,y_class,y_true,c_pred,c_true)
    
    # Get R^2 score among different groups
    m_TP = (y_true==1)&(y_class==1)
    m_TP_FP = (y_class==1)
    m_TP_FN = (y_true==1)
    
    if np.sum(m_TP) > 0:
        results_dict['Rsq_TP'] = metrics.r2_score(c_true[m_TP],c_pred[m_TP])
        results_dict['MAE_TP'] = metrics.mean_absolute_error(c_true[m_TP],c_pred[m_TP])
    else:
        results_dict['Rsq_TP'] = np.nan
        results_dict['MAE_TP'] = np.nan
    if np.sum(m_TP_FP) > 0:
        results_dict['Rsq_TP_FP'] = metrics.r2_score(c_true[m_TP_FP],c_pred[m_TP_FP])
        results_dict['MAE_TP_FP'] = metrics.mean_absolute_error(c_true[m_TP_FP],c_pred[m_TP_FP])
    else:
        results_dict['Rsq_TP_FP'] = np.nan
        results_dict['MAE_TP_FP'] = np.nan
        
    if np.sum(m_TP_FN) > 0:
        results_dict['Rsq_TP_FN'] = metrics.r2_score(c_true[m_TP_FN],c_pred[m_TP_FN])
        results_dict['MAE_TP_FN'] = metrics.mean_absolute_error(c_true[m_TP_FN],c_pred[m_TP_FN])
    else:
        results_dict['Rsq_TP_FN'] = np.nan
        results_dict['MAE_TP_FN'] = np.nan
    
    return(pd.DataFrame({'metric':results_dict.keys(),'value':results_dict.values()}))

def stratify_performance_by_sfha_pa(pa_predictions):
    
    nonpa_predictions = pa_predictions[~pa_predictions['pseudo_absence']]
    
    pa_sfha_predictions = pa_predictions[pa_predictions['SFHA']==1]
    pa_nonsfha_predictions = pa_predictions[pa_predictions['SFHA']==0]
    nonpa_sfha_predictions = nonpa_predictions[nonpa_predictions['SFHA']==1]
    nonpa_nonsfha_predictions = nonpa_predictions[nonpa_predictions['SFHA']==0]
    
    pa_overall_perf = get_performance(pa_predictions)
    pa_sfha_perf = get_performance(pa_sfha_predictions)
    pa_nonsfha_perf = get_performance(pa_nonsfha_predictions)
    
    nonpa_overall_perf = get_performance(nonpa_predictions)
    nonpa_sfha_perf = get_performance(nonpa_sfha_predictions)
    nonpa_nonsfha_perf = get_performance(nonpa_nonsfha_predictions)
    
    pa_overall_perf['pseudo_absence'] = 'included'
    pa_sfha_perf['pseudo_absence'] = 'included'
    pa_nonsfha_perf['pseudo_absence'] = 'included'
    
    nonpa_overall_perf['pseudo_absence'] = 'excluded'
    nonpa_sfha_perf['pseudo_absence'] = 'excluded'
    nonpa_nonsfha_perf['pseudo_absence'] = 'excluded'
    
    pa_overall_perf['SFHA'] = 'both'
    nonpa_overall_perf['SFHA'] = 'both'
    
    pa_sfha_perf['SFHA'] = 'inside'
    nonpa_sfha_perf['SFHA'] = 'inside'
    
    pa_nonsfha_perf['SFHA'] = 'outside'
    nonpa_nonsfha_perf['SFHA'] = 'outside'
    
    df = pd.concat([pa_overall_perf,pa_sfha_perf,pa_nonsfha_perf,nonpa_overall_perf,nonpa_sfha_perf,nonpa_nonsfha_perf]).reset_index(drop=True)
    
    return(df)


## Estimate model performance metrics across different stratifications:
## by event, including/excluding pseudo-absences, random/spatial cv, inside/outside SFHA

cv_perf_list = []

for i in range(len(floodevent_list)):

    random_cv_perf = stratify_performance_by_sfha_pa(floodevent_list[i].random_cv_predictions)
    spatial_cv_perf = stratify_performance_by_sfha_pa(floodevent_list[i].spatial_cv_predictions)

    random_cv_perf['cv_type'] = 'random'
    spatial_cv_perf['cv_type'] = 'spatial'

    event_perf = pd.concat([random_cv_perf,spatial_cv_perf])
    event_perf['event_year'] = floodevent_years[i]
    event_perf['event_name'] = floodevent_names[i]
    
    cv_perf_list.append(event_perf)

cv_perf = pd.concat(cv_perf_list).reset_index(drop=True)
cv_perf = cv_perf[['event_year','event_name','cv_type','pseudo_absence','SFHA','metric','value']]
cv_perf = pd.pivot(cv_perf,index=['event_year','event_name','cv_type','pseudo_absence','SFHA'],columns='metric',values='value')

outname = os.path.join(outfolder,'damage_prediction_cross_validation_performance.csv')
cv_perf.to_csv(outname)

# Helper function to create barplot of model performance metrics

def metrics_barplot(perf_data,metrics,metric_labels,metric_min,metric_max,metric_increment,cv_type='random',pseudo_absence='included',figsize=(8,10)):
    
    fig,axes = plt.subplots(nrows=len(metrics),ncols=1,figsize=figsize)

    major_spacing = 4
    minor_spacing = 1

    for i in range(len(metrics)):

        metric = metrics[i]
        metric_label = metric_labels[i]
        min_val = metric_min[i]
        max_val = metric_max[i]
        increment = metric_increment[i]

        ax = axes[i]

        y_overall = perf_data.loc[:,:,cv_type,pseudo_absence,'both'][metric]
        y_sfha = perf_data.loc[:,:,cv_type,pseudo_absence,'inside'][metric]
        y_nonsfha = perf_data.loc[:,:,cv_type,pseudo_absence,'outside'][metric]

        event_labels = [f'{x[1]}\n({x[0]})' for x in y_overall.index.values]
        n_events = len(event_labels)

        x_pos_sfha = np.arange(0,major_spacing*n_events,major_spacing)
        x_pos_overall = x_pos_sfha - minor_spacing
        x_pos_nonsfha = x_pos_sfha + minor_spacing

        alpha=1

        ax.bar(x_pos_overall,y_overall,width=minor_spacing,color='#004056',label='Overall',lw=1.5,edgecolor='k',alpha=alpha)
        ax.bar(x_pos_sfha,y_sfha,width=minor_spacing,color='#2C858D',label='SFHA',lw=1.5,edgecolor='k',alpha=alpha)
        ax.bar(x_pos_nonsfha,y_nonsfha,width=minor_spacing,color='#74CEB7',label='non-SFHA',lw=1.5,edgecolor='k',alpha=alpha)

        ax.set_xticks(x_pos_sfha)
        ax.set_xticklabels(event_labels)

        ax.set_yticks(np.arange(min_val,max_val+0.0001,increment))
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray')

        ax.set_ylim([min_val,max_val])
        
        if i==0:
            ax.legend(loc='upper left')

        ax.set_ylabel(metric_label,fontsize=12)

    fig.tight_layout()
    
    return(fig,axes)


# *** FIGURES S2-S4 *** ###

# Create barplots of presence-absence model performance
metric_names = ['roc_auc','accuracy','sensitivity','specificity','precision']
metric_labels = ['ROC AUC Score','Accuracy','Sensitivity','Specificity','Precision']
metric_min = [0.5,0.0,0.0,0.0,0.0]
metric_max = [1.0,1.0,1.0,1.0,1.0]
metric_increment = [0.05,0.1,0.1,0.1,0.1]

# Including pseudo-absences
fig,axes=metrics_barplot(cv_perf,metric_names,metric_labels,metric_min,metric_max,metric_increment,cv_type='random',pseudo_absence='included')
fig.tight_layout()
outname = os.path.join(outfolder,'Figure_S2_random_cv_performance_including_pseudo_absences.png')
fig.savefig(outname,dpi=400)

# Excluding pseudo-absences
fig,axes=metrics_barplot(cv_perf,metric_names,metric_labels,metric_min,metric_max,metric_increment,cv_type='random',pseudo_absence='excluded')
fig.tight_layout()
outname = os.path.join(outfolder,'Figure_S3_random_cv_performance_excluding_pseudo_absences.png')
fig.savefig(outname,dpi=400)

# Spatial CV
fig,axes=metrics_barplot(cv_perf,metric_names,metric_labels,metric_min,metric_max,metric_increment,cv_type='spatial',pseudo_absence='included')
fig.tight_layout()
outname = os.path.join(outfolder,'Figure_S4_spatial_cv_performance_including_pseudo_absences.png')
fig.savefig(outname,dpi=400)


### *** Figure S5 *** ###

n_events = len(floodevent_list)

fig,axes = plt.subplots(nrows=2,ncols=4,figsize=(11,6))

for i in range(n_events):
    
    ai = i // 4
    aj = i - 4*ai
    
    ax = axes[ai,aj]
    
    predictions = floodevent_list[i].random_cv_predictions
    
    geographic_units = floodevent_list[i].spatial_cv_tiles.rename(columns={'fold':'tile'})
    geographic_column = 'tile'

    predictions = gpd.sjoin_nearest(predictions,geographic_units,max_distance=1000).drop(columns=['index_right'])
    df = predictions[['total_cost','total_cost_pred',geographic_column]].groupby(geographic_column).sum().reset_index()

    x = df['total_cost_pred'].to_numpy()/1e6
    y = df['total_cost'].to_numpy()/1e6
    
    maxval = np.max(np.concatenate((x,y)))
        
    if maxval > 40:
        increment = 10
    elif maxval > 20:
        increment = 5
    elif maxval > 10:
        increment = 2
    else:
        increment = 1
    
    ticks = np.arange(0,maxval+increment,increment)
    
    minval = -0.05*maxval
    maxval = 1.05*maxval
    
    xline = np.linspace(minval,np.max(ticks),101)

    rsq = metrics.r2_score(y,x)
    yline = 1*xline
    
    ax.scatter(x,y,alpha=0.5,s=10)
    ax.plot(xline,yline,'k--',alpha=0.3)

    ax.set_ylim([minval,np.max(ticks)])
    ax.set_xlim([minval,np.max(ticks)])
    
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    ax.set_xlabel('Predicted damage ($M)')
    ax.set_ylabel('Observed damage ($M)')
    
    ax.text(0.8, 0.1, f'R²={rsq:.2f}',
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax.transAxes)
    
    title = f'{floodevent_names[i]} ({floodevent_years[i]})'
    ax.set_title(title,fontweight='bold')
    ax.set_aspect('equal')
    
axes[-1,-1].axis('off')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S5_damage_cost_Rsq_spatial_agg.png')
fig.savefig(outname,dpi=400)


### *** Similar figure for spatial CV *** ###

n_events = len(floodevent_list)

fig,axes = plt.subplots(nrows=2,ncols=4,figsize=(11,6))

for i in range(n_events):
    
    ai = i // 4
    aj = i - 4*ai
    
    ax = axes[ai,aj]
    
    predictions = floodevent_list[i].spatial_cv_predictions
    
    geographic_units = floodevent_list[i].spatial_cv_tiles.rename(columns={'fold':'tile'})
    geographic_column = 'tile'

    predictions = gpd.sjoin_nearest(predictions,geographic_units,max_distance=1000).drop(columns=['index_right'])
    df = predictions[['total_cost','total_cost_pred',geographic_column]].groupby(geographic_column).sum().reset_index()

    x = df['total_cost_pred'].to_numpy()/1e6
    y = df['total_cost'].to_numpy()/1e6
    
    maxval = np.max(np.concatenate((x,y)))
        
    if maxval > 40:
        increment = 10
    elif maxval > 20:
        increment = 5
    elif maxval > 10:
        increment = 2
    else:
        increment = 1
    
    ticks = np.arange(0,maxval+increment,increment)
    
    minval = -0.05*maxval
    maxval = 1.05*maxval
    
    xline = np.linspace(minval,np.max(ticks),101)

    rsq = metrics.r2_score(y,x)
    yline = 1*xline
    
    ax.scatter(x,y,alpha=0.5,s=10)
    ax.plot(xline,yline,'k--',alpha=0.3)

    ax.set_ylim([minval,np.max(ticks)])
    ax.set_xlim([minval,np.max(ticks)])
    
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    ax.set_xlabel('Predicted damage ($M)')
    ax.set_ylabel('Observed damage ($M)')
    
    ax.text(0.8, 0.1, f'R²={rsq:.2f}',
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax.transAxes)
    
    title = f'{floodevent_names[i]} ({floodevent_years[i]})'
    ax.set_title(title,fontweight='bold')
    ax.set_aspect('equal')
    
axes[-1,-1].axis('off')

fig.tight_layout()

outname = os.path.join(outfolder,'spatial_cv_damage_cost_Rsq_spatial_agg.png')
fig.savefig(outname,dpi=400)


### *** TABLE S2 *** ###

n_claims_list = []
n_address_policies_list = []
n_auxiliary_policies_list = []
n_presence_list = []
n_absence_list = []
n_pseudo_absence_list = []

for i in range(7):
    
    floodevent = floodevent_list[i]

    n_claims = len(floodevent.claims)
    n_address_policies = len(floodevent.policies)
    n_auxiliary_policies = len(floodevent.auxiliary_policies)

    n_presence = np.sum(floodevent.training_dataset['flood_damage']==1)
    n_absence = np.sum(floodevent.training_dataset['flood_damage']==0)
    n_pseudo_absence = floodevent.adjusted_training_dataset['pseudo_absence'].sum()
    
    n_claims_list.append(n_claims)
    n_address_policies_list.append(n_address_policies)
    n_auxiliary_policies_list.append(n_auxiliary_policies)
    n_presence_list.append(n_presence)
    n_absence_list.append(n_absence)
    n_pseudo_absence_list.append(n_pseudo_absence)
    
record_counts = pd.DataFrame(data={'event':floodevent_names,
                                   'n_claims':n_claims_list,
                                   'n_address_policies':n_address_policies_list,
                                   'n_auxiliary_policies':n_auxiliary_policies_list,
                                   'n_presence':n_presence_list,
                                   'n_absence':n_absence_list,
                                   'n_pseudo_absence':n_pseudo_absence_list})

outname = os.path.join(outfolder,'Table_S2_record_counts.csv')
record_counts.to_csv(outname,index=False)

# Compare claim totals to OpenFEMA records
n_total_claims = record_counts['n_claims'].sum()
n_total_policies = record_counts['n_address_policies'].sum()

openfema_claims_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/OpenFEMA/NC_FemaNfipClaims_v2.csv'
openfema_claims = pd.read_csv(openfema_claims_path)
openfema_claims['yearOfLoss'] = pd.to_datetime(openfema_claims['dateOfLoss']).dt.year
m = (openfema_claims['yearOfLoss'] >= 1996)&(openfema_claims['yearOfLoss'] <= 2019)
openfema_claims = openfema_claims[m]
openfema_claims['DamageAmount'] = openfema_claims['buildingDamageAmount'].fillna(0) + openfema_claims['contentsDamageAmount'].fillna(0)
openfema_claims = openfema_claims[openfema_claims['DamageAmount'] > 0]

n_openfema_claims = len(openfema_claims)

print('Number of included claims:',n_total_claims,flush=True)
print('Number of included policies:',n_total_policies,flush=True)
print('Number of OpenFEMA claims in NC, 1996-2019:',n_openfema_claims,flush=True)



### *** Figure S6 *** ###

## Property value estimation cross-validation error
property_value_dir = '/proj/characklab/flooddata/NC/multiple_events/analysis/property_value_estimates_by_county'
complete_counties = [x.name for x in os.scandir(property_value_dir) if x.is_dir() and not x.name.startswith('.')]
complete_counties = np.sort(complete_counties)

cv_list = []

for county in study_area_counties:
        
    cv_filepath = os.path.join(property_value_dir,f'{county}/{county}_cross_validation.parquet')
    
    # Read in cross-validation data
    table = pq.read_table(cv_filepath,use_pandas_metadata=True)
    cv_list.append(table.to_pandas())
    
cv_df = pd.concat(cv_list).reset_index(drop=True)
cv_df['year'] = cv_df['date_transfer'].dt.year

# Denote lowest 5% of property value transactions in each year
low_pv_transactions = cv_df[['val_transfer','year']].groupby('year').quantile(0.1).reset_index().rename(columns={'val_transfer':'P10_val_transfer'})
cv_df = pd.merge(cv_df,low_pv_transactions,on='year',how='left')



# Get distirbution of relative error

m = (cv_df['val_transfer'] > cv_df['P10_val_transfer'])

d1 = dm.empirical_distribution(cv_df['abs_percent_error_nonlog'].to_numpy(),estimate_pdf=False)
d2 = dm.empirical_distribution(cv_df[m]['abs_percent_error_nonlog'].to_numpy(),estimate_pdf=False)

xvals = np.linspace(0,1.2,1000)
y1vals = d1.cdf(xvals)
y2vals = d2.cdf(xvals)

xvals *= 100
y1vals *= 100
y2vals *= 100


fig,ax = plt.subplots(figsize=(6,4))

xticks = np.arange(0,120+1,20)
xticklabels = [f'{int(x)}%' for x in xticks]

yticks = np.arange(0,100+1,10)
yticklabels = [f'{int(y)}%' for y in yticks]

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.set_xlabel('Absolute percentage error tolerance, %')
ax.set_ylabel('Proportion of sales within tolerance, %')

ax.plot(xvals,y1vals,lw=2,color='C0',label='All single-family homes')
ax.plot(xvals,y2vals,lw=2,color='C3',label='Excluding lowest 10% of\nhome sales in each year')

ax.grid('on')

ax.set_xlim([0,120])
ax.set_ylim([0,100])

ax.legend()

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S6_property_value_error_CDF.png')
fig.savefig(outname,dpi=400)

fig.show()

print('All:')
print(np.round(100*d1.cdf(0.2)),'within 20%')
print(np.round(100*d1.cdf(0.5)),'within 50%')

print('Excluding lowest 10%:')
print(np.round(100*d2.cdf(0.2)),'within 20%')
print(np.round(100*d2.cdf(0.5)),'within 50%')



### *** FIGURE S7 *** ###
years = np.arange(1990,2019+1)

error_by_year = []

for year in years:
    error_by_year.append(cv_df[cv_df['year']==year]['abs_error_nonlog'].to_numpy())
    
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,4))

ax.boxplot(error_by_year,whis=(10,90),sym='',medianprops={'color':'k'})
ax.set_xticklabels(years,rotation=45)

ymax = 225000
yticks = np.arange(0,ymax+1,25000)
yticklabels = [f'${int(y/1000)}k' for y in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_ylim(0,ymax)

ax2 = ax.twinx()
ax2.set_yticks(yticks)
ax2.set_yticks(yticks)
ax2.set_yticklabels(yticklabels,color='C0',alpha=0.75)
ax2.set_ylim(0,ymax)

ax2.set_ylabel('Median sale price, USD',color='C0',alpha=0.75)

ax.set_ylabel('Absolute error, USD')

xpos = np.arange(len(years))+1
median_sale_price = cv_df[['val_transfer','year']].groupby('year').median()['val_transfer'].to_numpy()

ax.plot(xpos,median_sale_price,color='C0',lw=2,linestyle='-',alpha=0.75,zorder=-2)

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S7_property_value_error_by_year.png')
fig.savefig(outname,dpi=400)

fig.show()



### *** FIGURE S8 *** ###

error_by_county = cv_df[['countyCode','abs_percent_error_nonlog']].groupby('countyCode').median().reset_index()
error_by_county = pd.merge(counties,error_by_county,on='countyCode',how='left')
error_by_county['abs_percent_error_nonlog'] *= 100

fig,ax = plt.subplots(figsize=(7,5))

missing_kwds={'edgecolor':'k','facecolor':'grey'}
legend_kwds = {'label':'Median absolute percentage error, %','orientation':'horizontal'}

cmap = plt.get_cmap('Reds', 10)
error_by_county.plot(ax=ax,column='abs_percent_error_nonlog',cmap=cmap,vmin=0,vmax=100,edgecolor='k',missing_kwds=missing_kwds,legend=True,legend_kwds=legend_kwds)

ax.set_xlabel('UTM Zone 17N x-coord, m')
ax.set_ylabel('UTM Zone 17N y-coord, m')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S8_property_value_error_by_county.png')
fig.savefig(outname,dpi=400)

fig.show()


m = ~error_by_county['abs_percent_error_nonlog'].isna()
n_less = (error_by_county[m]['abs_percent_error_nonlog'] <= 50).sum()
n_tot = np.sum(m)
print(f'{n_less} / {n_tot}')


### Read in data from borrower simulation model ###

mortgage_sim_dir = '/proj/characklab/flooddata/NC/multiple_events/analysis/mortgage_borrower_simulation_base_case_postprocessed'

sim_filepath = os.path.join(mortgage_sim_dir,'simulation_output_damaged.parquet')
sim_df = pq.read_table(sim_filepath,use_pandas_metadata=True).to_pandas()

# Create income categories
income_quantiles = pd.read_csv(os.path.join(mortgage_sim_dir,'income_quantiles.csv'))
income_quantiles['period'] = pd.to_datetime(income_quantiles['period']).dt.to_period('M')
income_quantiles = income_quantiles.set_index('period')
income_quantiles['P0'] = 0.0
income_quantiles['P100'] = np.inf
income_quantiles = income_quantiles[['P0','P20','P40','P60','P80','P100']]

income_binning_function = lambda x: np.digitize(x['monthly_income'],bins=income_quantiles.loc[x['period']].values)

sim_df['income_group'] = sim_df.apply(income_binning_function,axis=1)

# Create property value categories
pv_quantiles = pd.read_csv(os.path.join(mortgage_sim_dir,'property_value_quantiles.csv'))
pv_quantiles['period'] = pd.to_datetime(pv_quantiles['period']).dt.to_period('M')
pv_quantiles = pv_quantiles.set_index('period')
pv_quantiles['P0'] = 0.0
pv_quantiles['P100'] = np.inf
pv_quantiles = pv_quantiles[['P0','P20','P40','P60','P80','P100']]

pv_binning_function = lambda x: np.digitize(x['property_value'],bins=pv_quantiles.loc[x['period']].values)

sim_df['property_value_group'] = sim_df.apply(pv_binning_function,axis=1)



### *** FIGURE S9 *** ###

# Read in data on survival of simulated loans
simulated_surv = pd.read_csv(os.path.join(mortgage_sim_dir,'simulated_survival.csv'))
t = simulated_surv['loan_age']/12

# Read in data on empirically-observed survival of loans by GSEs during 1999-2021 period
surv_dir = os.path.join(pwd,'2024-03-31_loan_survival_analysis')

p30_km_gse = pd.read_csv(os.path.join(surv_dir,'purchase30_km_estimate.csv'))
r30_km_gse = pd.read_csv(os.path.join(surv_dir,'refinance30_km_estimate.csv'))
r15_km_gse = pd.read_csv(os.path.join(surv_dir,'refinance15_km_estimate.csv'))

# For 15-year refinance, cut off at loan term since there's a small number of loans that stay on books despite being years delinquent
r15_km_gse = r15_km_gse[r15_km_gse['time'] <= 180]
simulated_surv.loc[simulated_surv['loan_age'] > 180,'r15_surv'] = np.nan

ft=12
yticks = np.arange(0,1+0.01,0.25)
xticks = np.arange(0,30+1,5)

fig,axes = plt.subplots(nrows=3,ncols=1,figsize=(6,6))

ax = axes[0]

ax.plot(p30_km_gse['time']/12,p30_km_gse['surv'],color='C0',lw=2,label='GSE data')
ax.plot(t,simulated_surv['p30_surv'],color='C3',lw=2,label='Simulated',ls='-.')
ax.set_xlim([0,30])
ax.set_ylim([0,1])
ax.set_title('Home purchase: 30-year term',fontsize=ft,weight='bold')
#ax.set_xlabel('Loan age, years')
ax.set_ylabel('Survival probability')
ax.set_yticks(yticks)
ax.set_xticks(xticks)
ax.legend()
ax.grid('on')

ax = axes[1]

ax.plot(r30_km_gse['time']/12,r30_km_gse['surv'],color='C0',lw=2,label='GSE data')
ax.plot(t,simulated_surv['r30_surv'],color='C3',lw=2,label='Simulated',ls='-.')
ax.set_xlim([0,30])
ax.set_ylim([0,1])
ax.set_title('Refinance: 30-year term',fontsize=ft,weight='bold')
#ax.set_xlabel('Loan age, years')
ax.set_ylabel('Survival probability')
ax.set_yticks(yticks)
ax.set_xticks(xticks)
#ax.legend()
ax.grid('on')

ax = axes[2]

ax.plot(r15_km_gse['time']/12,r15_km_gse['surv'],color='C0',lw=2,label='GSE data')
ax.plot(t,simulated_surv['r15_surv'],color='C3',lw=2,label='Simulated',ls='-.')
ax.set_xlim([0,30])
ax.set_ylim([0,1])
ax.set_title('Refinance: 15-year term',fontsize=ft,weight='bold')
ax.set_xlabel('Loan age, years')
ax.set_ylabel('Survival probability')
ax.set_yticks(yticks)
ax.set_xticks(xticks)
#ax.legend()
ax.grid('on')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S9_Kaplan-Meier.png')
fig.savefig(outname,dpi=400)

fig.show()



### Read in post-processed data on damages

filepath = '/proj/characklab/flooddata/NC/multiple_events/analysis/2024-09-11_damage_estimates_by_county/statewide_flood_damage_exposure.parquet'
exposure_df = pq.read_table(filepath).to_pandas().rename(columns={'censusTract':'censusTract_2010'}).drop(columns=['count_times_flooded','repetitive'])

# Get building-timepoint observations in which flood damage occurs
exposure_df = exposure_df[exposure_df['flood_damage']==1].sort_values(by='period')

# Categorize damage as repetitive or non-repetitive
exposure_df['count_times_flooded'] = exposure_df.groupby('building_id').cumcount() + 1
exposure_df['repetitive'] = (exposure_df['count_times_flooded'] >= 2).astype(int)
exposure_df['nonrepetitive'] = 1-exposure_df['repetitive']

# Read in info on number of buildings in different areas (e.g., urban/rural, SFHA/non-SFHA, etc.)
structure_info_filepath = '/proj/characklab/flooddata/NC/multiple_events/analysis/building_counts/structure_info.parquet'
structure_info = pd.read_parquet(structure_info_filepath).rename(columns={'single_family':'single_family_detached'})
structure_info = structure_info[structure_info['countyName'].isin(study_area_counties)].reset_index(drop=True)
CAMA_counties = structure_info[structure_info['CAMA']==1]['countyName'].unique()

# Attach info on CAMA / urban to exposure dataframe
exposure_df = pd.merge(exposure_df,structure_info[['building_id','CAMA','urban']],how='left',on='building_id')
exposure_df['year'] = exposure_df['period'].dt.year

# Attach info on return periods (<100y,100-500y,>500y) and distance to SFHA
filepath = '/proj/characklab/flooddata/NC/multiple_events/analysis/evaluate_exposure_outside_SFHA/buildings_floodzone_detail.parquet'
floodzone_detail = pq.read_table(filepath).to_pandas()
exposure_df = pd.merge(exposure_df,floodzone_detail,how='left',on='building_id')
exposure_df['dist_SFHA_category'] = pd.cut(exposure_df['dist_SFHA'],[-np.inf,0,250,np.inf],labels=['d <= 0','0 < d <= 250','d > 250'])




# Get counts of buildings by county / SFHA / urban
count_by_county = gpd.read_file('/proj/characklab/flooddata/NC/multiple_events/analysis/building_counts/counties')
count_by_county_SFHA = pd.read_parquet('/proj/characklab/flooddata/NC/multiple_events/analysis/building_counts/count_by_county_SFHA.parquet')
count_by_county_urban = pd.read_parquet('/proj/characklab/flooddata/NC/multiple_events/analysis/building_counts/count_by_county_urban.parquet')

count_by_county = count_by_county[count_by_county['countyName'].isin(study_area_counties)]
count_by_county_SFHA = count_by_county_SFHA[count_by_county_SFHA['countyName'].isin(study_area_counties)]
count_by_county_urban = count_by_county_urban[count_by_county_urban['countyName'].isin(study_area_counties)]

count_by_SFHA = count_by_county_SFHA.groupby('SFHA').sum().drop(columns='countyName').reset_index()
count_by_urban = count_by_county_urban.groupby('urban').sum().drop(columns='countyName').reset_index()

count_by_county_CAMA = count_by_county.copy()
count_by_county_CAMA['CAMA'] = count_by_county['countyName'].isin(CAMA_counties).astype(int)
count_by_CAMA = count_by_county_CAMA.drop(columns=['countyName','geometry']).groupby('CAMA').sum().reset_index()

count_by_CAMA_SFHA = structure_info[['CAMA','SFHA','building_id','single_family_detached']].groupby(['CAMA','SFHA']).agg({'building_id':'count','single_family_detached':'sum'}).reset_index()
count_by_CAMA_SFHA = count_by_CAMA_SFHA.rename(columns={'building_id':'bldg_count','single_family_detached':'sf_count'})



exposure_df['count_buildings_flooded'] = exposure_df['building_id']
exposure_df['total_real_cost'] = exposure_df['uninsured_real_cost'] + exposure_df['insured_real_cost']
exposure_df['UNR'] = exposure_df['uninsured_real_cost']*exposure_df['nonrepetitive']
exposure_df['UR'] = exposure_df['uninsured_real_cost']*exposure_df['repetitive']
exposure_df['IR'] = exposure_df['insured_real_cost']*exposure_df['repetitive']
exposure_df['INR'] = exposure_df['insured_real_cost']*exposure_df['nonrepetitive']

# Note: Because a given building can flood multiple times over the study period, 'count_times_flooded' and 
# 'count_buildings_flooded' may be different depending on the variables used for aggregation.  

exp_columns = ['count_times_flooded',
               'count_buildings_flooded',
               'insured_real_cost',
               'uninsured_real_cost',
               'total_real_cost',
               'UNR',
               'UR',
               'IR',
               'INR']

building_agg_func = lambda x: len(np.unique(x))

agg_dict = {'count_times_flooded':'count',
            'count_buildings_flooded':building_agg_func,
            'insured_real_cost':'sum',
            'uninsured_real_cost':'sum',
            'total_real_cost':'sum',
            'UNR':'sum',
            'UR':'sum',
            'IR':'sum',
            'INR':'sum'}

exp_by_event = exposure_df[['year'] + exp_columns].groupby('year').agg(agg_dict).reset_index()
exp_by_event_county = exposure_df[['year','countyName'] + exp_columns].groupby(['year','countyName']).agg(agg_dict).reset_index()
exp_by_event_SFHA = exposure_df[['year','SFHA'] + exp_columns].groupby(['year','SFHA']).agg(agg_dict).reset_index()
exp_by_event_CAMA = exposure_df[['year','CAMA'] + exp_columns].groupby(['year','CAMA']).agg(agg_dict).reset_index()
exp_by_event_urban = exposure_df[['year','urban'] + exp_columns].groupby(['year','urban']).agg(agg_dict).reset_index()

exp_by_county = exposure_df[['countyName'] + exp_columns].groupby('countyName').agg(agg_dict).reset_index()
exp_by_SFHA = exposure_df[['SFHA'] + exp_columns].groupby(['SFHA']).agg(agg_dict).reset_index()
exp_by_CAMA = exposure_df[['CAMA'] + exp_columns].groupby(['CAMA']).agg(agg_dict).reset_index()
exp_by_urban = exposure_df[['urban'] + exp_columns].groupby(['urban']).agg(agg_dict).reset_index()

exp_by_CAMA_SFHA = exposure_df[['CAMA','SFHA'] + exp_columns].groupby(['CAMA','SFHA']).agg(agg_dict).reset_index()



# See how many flood-exposed buildings inside/outside the SFHA can be identified 
# based on insurance records alone. Later we'll compare this to estimates that

exp_by_zonedetail = exposure_df[['floodzone_detail'] + exp_columns].groupby(['floodzone_detail']).agg(agg_dict).reset_index()
exp_by_zonedetail.to_csv(os.path.join(outfolder,'exp_by_zonedetail.csv'),index=False)

# include damage to uninsured households. 
m = (exposure_df['insured']==1)
exp_by_zonedetail_insured = exposure_df[m][['floodzone_detail'] + exp_columns].groupby(['floodzone_detail']).agg(agg_dict).reset_index()
exp_by_zonedetail_insured.to_csv(os.path.join(outfolder,'insured_exp_by_zonedetail.csv'),index=False)

# See how many flood-exposed buildings within different distance from the SFHA can be identified 

exp_by_distcat = exposure_df[['dist_SFHA_category'] + exp_columns].groupby(['dist_SFHA_category']).agg(agg_dict).reset_index()
exp_by_distcat.to_csv(os.path.join(outfolder,'exp_by_distcat.csv'),index=False)

# include damage to uninsured households.
m = (exposure_df['insured']==1)
exp_by_distcat_insured = exposure_df[m][['dist_SFHA_category'] + exp_columns].groupby(['dist_SFHA_category']).agg(agg_dict).reset_index()
exp_by_distcat_insured.to_csv(os.path.join(outfolder,'insured_exp_by_distcat.csv'),index=False)


# Add count of total number of unique buildings (exposed + unexposed) in each grouping
exp_by_event['bldg_count'] = len(structure_info)
exp_by_event_county = pd.merge(exp_by_event_county,count_by_county[['countyName','bldg_count']])
exp_by_event_SFHA = pd.merge(exp_by_event_SFHA,count_by_SFHA[['SFHA','bldg_count']])
exp_by_event_CAMA = pd.merge(exp_by_event_CAMA,count_by_CAMA[['CAMA','bldg_count']])
exp_by_event_urban = pd.merge(exp_by_event_urban,count_by_urban[['urban','bldg_count']])
exp_by_county = pd.merge(exp_by_county,count_by_county[['countyName','bldg_count']])
exp_by_SFHA = pd.merge(exp_by_SFHA,count_by_SFHA[['SFHA','bldg_count']])
exp_by_CAMA = pd.merge(exp_by_CAMA,count_by_CAMA[['CAMA','bldg_count']])
exp_by_urban = pd.merge(exp_by_urban,count_by_urban[['urban','bldg_count']])
exp_by_CAMA_SFHA = pd.merge(exp_by_CAMA_SFHA,count_by_CAMA_SFHA[['CAMA','SFHA','bldg_count']])


exp_by_event.to_csv(os.path.join(outfolder,'exp_by_event.csv'),index=False)
exp_by_event_county.to_csv(os.path.join(outfolder,'exp_by_event_county.csv'),index=False)
exp_by_event_SFHA.to_csv(os.path.join(outfolder,'exp_by_event_SFHA.csv'),index=False)
exp_by_event_CAMA.to_csv(os.path.join(outfolder,'exp_by_event_CAMA.csv'),index=False)
exp_by_county.to_csv(os.path.join(outfolder,'exp_by_county.csv'),index=False)
exp_by_SFHA.to_csv(os.path.join(outfolder,'exp_by_SFHA.csv'),index=False)
exp_by_CAMA.to_csv(os.path.join(outfolder,'exp_by_CAMA.csv'),index=False)
exp_by_urban.to_csv(os.path.join(outfolder,'exp_by_urban.csv'),index=False)
exp_by_CAMA_SFHA.to_csv(os.path.join(outfolder,'exp_by_CAMA_SFHA.csv'),index=False)


# Get number of buildings flooded 1, 2, 3, or 4+ times by comparative group 

times_flooded = exposure_df[['building_id','count_times_flooded']].groupby('building_id').max().reset_index()
times_flooded = pd.merge(times_flooded,structure_info[['building_id','SFHA','CAMA','urban']])

times_flooded['f1'] = (times_flooded['count_times_flooded'] == 1).astype(int)
times_flooded['f2'] = (times_flooded['count_times_flooded'] == 2).astype(int)
times_flooded['f3'] = (times_flooded['count_times_flooded'] == 3).astype(int)
times_flooded['f4plus'] = (times_flooded['count_times_flooded'] >= 4).astype(int)
count_cols = ['f1','f2','f3','f4plus']
times_flooded = times_flooded[['SFHA','CAMA','urban'] + count_cols].groupby(['SFHA','CAMA','urban']).sum().reset_index()
times_flooded['ftotal'] = times_flooded[count_cols].sum(axis=1)
times_flooded = pd.merge(times_flooded,structure_info[['building_id','SFHA','CAMA','urban']].groupby(['SFHA','CAMA','urban']).count().rename(columns={'building_id':'bldg_count'}).reset_index())
times_flooded.to_csv(os.path.join(outfolder,'times_flooded_by_group.csv'),index=False)

count_cols = ['f1','f2','f3','f4plus','ftotal','bldg_count']

times_flooded_by_SFHA = times_flooded[['SFHA']+count_cols].groupby('SFHA').sum()
times_flooded_by_SFHA['prob'] = times_flooded_by_SFHA['ftotal']/times_flooded_by_SFHA['bldg_count']

times_flooded_by_CAMA = times_flooded[['CAMA']+count_cols].groupby('CAMA').sum()
times_flooded_by_CAMA['prob'] = times_flooded_by_CAMA['ftotal']/times_flooded_by_CAMA['bldg_count']

times_flooded_by_urban = times_flooded[['urban']+count_cols].groupby('urban').sum()
times_flooded_by_urban['prob'] = times_flooded_by_urban['ftotal']/times_flooded_by_urban['bldg_count']


### *** FIGURE 3 - PART I *** ###

alpha=0.7

normalize=1e9

U_color = 'C3'
I_color = 'C0'

hatch = '//'

exp_df = exp_by_event.copy()

xticklabels = ['Fran\n(1996)','Bonnie\n(1998)','Floyd\n(1999)','Isabel\n(2003)','Irene\n(2011)','Matthew\n(2016)','Florence\n(2018)']
xticks = np.arange(len(xticklabels))

UNR = exp_df.loc[:,'UNR'].to_numpy()/normalize
UR = exp_df.loc[:,'UR'].to_numpy()/normalize
U = UNR + UR

IR = exp_df.loc[:,'IR'].to_numpy()/normalize
INR = exp_df.loc[:,'INR'].to_numpy()/normalize
I = INR + IR

T = U+I

R = UR + IR

U_bottom = np.zeros(len(xticks))
I_bottom = U
R_bottom = UNR

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6.0,4))

bottom = np.zeros(len(xticks))

# Insured
ax.bar(xticks,I,bottom=I_bottom,color=I_color,alpha=alpha,ec='k',lw=1,label='Insured')

# Uninsured
ax.bar(xticks,U,bottom=U_bottom,color=U_color,alpha=alpha,ec='k',lw=1,label='Uninsured')

# Repetitive
ax.bar(xticks,R,bottom=R_bottom,facecolor='none',hatch=hatch,ec='k',lw=1,label='Repetitive')

# Add labels
pad = 0.05
for i in range(len(xticks)):
    
    ax.text(xticks[i],T[i]+pad,money_label(T[i]*normalize),ha='center',va='bottom')
    
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

ax.set_ylabel('Flood damage exposure, billions USD')

ax.set_ylim([0,1.75])
ax.set_yticks(np.arange(0,2,0.25))

ax.legend(loc='upper left')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_3_part_I_damage_cost_by_event.png')
fig.savefig(outname,dpi=400)

fig.show()


### *** FIGURE 3 - PART II *** ###

colors = [U_color,U_color,I_color,I_color]
hatch = '//'

wedgeprops={'alpha':alpha,'edgecolor':'k','linewidth':2.0}

x = [UNR.sum(),UR.sum(),IR.sum(),INR.sum()]

startangle=360*np.sum(x[1:3])/np.sum(x)*1.1

fig,ax = plt.subplots(figsize=(3,3))

wedges = ax.pie(x,colors=colors,startangle=startangle,wedgeprops=wedgeprops)

for pie_wedge in wedges[0][1:3]:
    pie_wedge.set_hatch(hatch)

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_3_part_II_damage_cost_by_category.png')
fig.savefig(outname,dpi=400)

fig.show()


### *** FIGURE 4 *** ###

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(8,4))

ax = axes[0]

## Part I: DAMAGE COST

# Get exposure by coastal / non-coastal, 
order = [0,1]
UNR_CAMA = exp_by_CAMA.loc[order,'UNR'].to_numpy()/normalize
UR_CAMA = exp_by_CAMA.loc[order,'UR'].to_numpy()/normalize
IR_CAMA = exp_by_CAMA.loc[order,'IR'].to_numpy()/normalize
INR_CAMA = exp_by_CAMA.loc[order,'INR'].to_numpy()/normalize

UNR_urban = exp_by_urban.loc[order,'UNR'].to_numpy()/normalize
UR_urban = exp_by_urban.loc[order,'UR'].to_numpy()/normalize
IR_urban = exp_by_urban.loc[order,'IR'].to_numpy()/normalize
INR_urban = exp_by_urban.loc[order,'INR'].to_numpy()/normalize

UNR_SFHA = exp_by_SFHA.loc[order,'UNR'].to_numpy()/normalize
UR_SFHA = exp_by_SFHA.loc[order,'UR'].to_numpy()/normalize
IR_SFHA = exp_by_SFHA.loc[order,'IR'].to_numpy()/normalize
INR_SFHA = exp_by_SFHA.loc[order,'INR'].to_numpy()/normalize

UNR = np.concatenate((UNR_urban,UNR_CAMA,UNR_SFHA))
UR = np.concatenate((UR_urban,UR_CAMA,UR_SFHA))
IR = np.concatenate((IR_urban,IR_CAMA,IR_SFHA))
INR = np.concatenate((INR_urban,INR_CAMA,INR_SFHA))


U = UNR + UR
I = INR + IR
T = U+I
R = UR + IR

U_left = np.zeros(len(UNR))
I_left = U
R_left = UNR

yticks = [0,1,3,4,6,7]
yticklabels = ['Rural','Urban','Inland','Coastal','Non-SFHA','SFHA']

bottom = np.zeros(len(xticks))

# Insured
ax.barh(yticks,I,left=I_left,color=I_color,alpha=alpha,ec='k',lw=1,label='Insured')

# Uninsured
ax.barh(yticks,U,left=U_left,color=U_color,alpha=alpha,ec='k',lw=1,label='Uninsured')

# Repetitive
ax.barh(yticks,R,left=R_left,facecolor='none',hatch=hatch,ec='k',lw=1,label='Repetitive')

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.175),ncol=3,fontsize=8)

ax.set_xlabel('Flood damage exposure, billions USD')


# PART II: NUMBER OF TIMES FLOODED

ax = axes[1]

order = [0,1]

f1_SFHA = times_flooded_by_SFHA.loc[order,'f1'].to_numpy()
f2_SFHA = times_flooded_by_SFHA.loc[order,'f2'].to_numpy()
f3_SFHA = times_flooded_by_SFHA.loc[order,'f3'].to_numpy()
f4_SFHA = times_flooded_by_SFHA.loc[order,'f4plus'].to_numpy()
prob_SFHA = times_flooded_by_SFHA.loc[order,'prob'].to_numpy()

f1_CAMA = times_flooded_by_CAMA.loc[order,'f1'].to_numpy()
f2_CAMA = times_flooded_by_CAMA.loc[order,'f2'].to_numpy()
f3_CAMA = times_flooded_by_CAMA.loc[order,'f3'].to_numpy()
f4_CAMA = times_flooded_by_CAMA.loc[order,'f4plus'].to_numpy()
prob_CAMA = times_flooded_by_CAMA.loc[order,'prob'].to_numpy()

f1_urban = times_flooded_by_urban.loc[order,'f1'].to_numpy()
f2_urban = times_flooded_by_urban.loc[order,'f2'].to_numpy()
f3_urban = times_flooded_by_urban.loc[order,'f3'].to_numpy()
f4_urban = times_flooded_by_urban.loc[order,'f4plus'].to_numpy()
prob_urban = times_flooded_by_urban.loc[order,'prob'].to_numpy()

f1 = np.concatenate((f1_urban,f1_CAMA,f1_SFHA))
f2 = np.concatenate((f2_urban,f2_CAMA,f2_SFHA))
f3 = np.concatenate((f3_urban,f3_CAMA,f3_SFHA))
f4 = np.concatenate((f4_urban,f4_CAMA,f4_SFHA))
prob = np.concatenate((prob_urban,prob_CAMA,prob_SFHA))

f1_left = np.zeros(len(f1))
f2_left = f1
f3_left = f1 + f2
f4_left = f1 + f2 + f3
prob_left = f1 + f2 + f3 + f4

cmap = plt.get_cmap('Blues', 6)
offset = 2
hex_codes = [mpl.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

yticks = [0,1,3,4,6,7]
yticklabels = ['Rural','Urban','Inland','Coastal','Non-SFHA','SFHA']

bottom = np.zeros(len(xticks))

# Flooded once
ax.barh(yticks,f1,left=f1_left,color=hex_codes[offset+0],ec='k',lw=0,label='1')

# Flooded twice
ax.barh(yticks,f2,left=f2_left,color=hex_codes[offset+1],ec='k',lw=0,label='2')

# Flooded three times
ax.barh(yticks,f3,left=f3_left,color=hex_codes[offset+2],ec='k',lw=0,label='3')

# Flooded four or more times
ax.barh(yticks,f4,left=f4_left,color=hex_codes[offset+3],ec='k',lw=0,label='≥4')

# Add labels
pad = 250
for i in range(len(yticks)):
    ax.text(prob_left[i]+prob[i]+pad,yticks[i],f'{100*prob[i]:.1f}%',ha='left',va='center',fontsize=8)

ax.set_yticks([])
#ax.set_yticks(yticks)
#ax.set_yticklabels(yticklabels)


ax.set_xlim([0,40000])

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.175),title='Number of times flooded',ncol=4,fontsize=8)

ax.set_xlabel('Number of flood-damaged structures')

# Add lettering to distinguish panels
letters = ['(a)','(b)']

for i,ax in enumerate(axes.flat):
    ax.text(0.93, 1.03, letters[i], transform=ax.transAxes, size=ft-2)

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_4_flood_exposure_by_comparative_group.png')
fig.savefig(outname,dpi=400)

fig.show()


### *** FIGURE 5 PRE-PROCESSING *** ###

hexagon_path = os.path.join(pwd,'building_counts/hexagons')
hexagons = gpd.read_file(hexagon_path)

hex_exp = exposure_df[exposure_df['total_real_cost'] > 0]
    
hex_exp = pd.merge(hex_exp,structure_info[['building_id','tile_id']],on='building_id',how='left')
hex_exp = hex_exp[['tile_id','UNR','UR','IR','INR']].groupby('tile_id').sum().reset_index()
hex_exp = pd.merge(hexagons,hex_exp,on='tile_id',how='left').fillna(0)
hex_exp.fillna(0)

hex_exp['I'] = hex_exp['INR'] + hex_exp['IR']
hex_exp['U'] = hex_exp['UNR'] + hex_exp['UR']
hex_exp['R'] = hex_exp['UR'] + hex_exp['IR']
hex_exp['T'] = hex_exp[['UNR','UR','IR','INR']].sum(axis=1)

hex_exp['I_M'] = hex_exp['I']/1e6
hex_exp['U_M'] = hex_exp['U']/1e6
hex_exp['R_M'] = hex_exp['R']/1e6

hex_exp = hex_exp[hex_exp['on_land']==1]

m_nodata = (hex_exp['bldg_count'] == 0)
hex_nodata = hex_exp[m_nodata]

num_years = 2020-1996

hex_exp['I_per'] = hex_exp['I']/(hex_exp['bldg_count']*num_years)
hex_exp['U_per'] = hex_exp['U']/(hex_exp['bldg_count']*num_years)
hex_exp['R_per'] = hex_exp['R']/(hex_exp['bldg_count']*num_years)
hex_exp['IR_per'] = hex_exp['IR']/(hex_exp['bldg_count']*num_years)
hex_exp['UR_per'] = hex_exp['UR']/(hex_exp['bldg_count']*num_years)
hex_exp['T_per'] = hex_exp['T']/(hex_exp['bldg_count']*num_years)

hex_exp = hex_exp[hex_exp['T_per']>0]

# Read in county COGs
COG_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/NCARCOGS/NCARCOG_regional_councils.xlsx'
COGs = pd.read_excel(COG_path,sheet_name='countyCOGs')

# Subset to Eastern NC region
included_COGs = ['Abermarle','Mid-East','Eastern Carolina','Cape Fear','Lumber River','Mid-Carolina','Central Pines','Upper Coastal Plain','Kerr-Tar']

COGs = COGs[COGs['NCARCOG'].isin(included_COGs)]

print('Number of included counties:',len(COGs))

# Read in counties with pretty coastline
counties_coast_path = '/proj/characklab/flooddata/NC/R_files/git_data/plot_data/CountyBoundaryShoreline/CountyBoundaryShoreline_SHP/BoundaryCountyShoreline.shp'
counties_coast = gpd.read_file(counties_coast_path).to_crs(crs)
intercoastal = counties_coast[counties_coast['WATER']=='S']
intercoastal = intercoastal.dissolve()
counties_coast = counties_coast[counties_coast['WATER']=='A']

# Subset to counties in included COGs
counties_coast = counties_coast[counties_coast['CountyName'].isin(COGs['countyName'])]

COG_domain = counties_coast.dissolve()['geometry'].values[0]

hex_exp = hex_exp[hex_exp.intersects(COG_domain)]

# Major hydro
hydro_path = '/proj/characklab/flooddata/NC/R_files/git_data/plot_data/NC_MajorHydro_rivers/NC_MajorHydro_rivers_edit.shp'
hydro = gpd.read_file(hydro_path).to_crs(crs)

hydro = hydro[hydro.intersects(COG_domain)]
hydro['geometry'] = hydro['geometry'].intersection(COG_domain)


### *** FIGURE 5: ANNUALIZED & CUMULATIVE COSTS *** ###

county_lw = 0.85
hydro_lw = 0.25

cmap1 = plt.get_cmap('GnBu')
norm1=mpl.colors.LogNorm(vmin=1, vmax=1000)

cmap2 = plt.get_cmap('RdPu')
norm2=mpl.colors.LogNorm(vmin=1, vmax=100)

fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(7,7))

ax = axes[0,0]

hex_exp.plot(ax=ax,column='I_per',cmap=cmap1,norm=norm1,alpha=1)
counties_coast.plot(ax=ax,facecolor='none',edgecolor='k',alpha=1,lw=county_lw)
hydro.plot(ax=ax,color='k',lw=hydro_lw)
ax.set_ylabel('Insured damage',weight='bold',fontsize=ft+2)

ax.set_xticks([])
ax.set_yticks([])

ax = axes[0,1]

hex_exp.plot(ax=ax,column='I_M',cmap=cmap2,norm=norm2,alpha=1)
counties_coast.plot(ax=ax,facecolor='none',edgecolor='k',alpha=1,lw=county_lw)
hydro.plot(ax=ax,color='k',lw=hydro_lw)

ax.set_xticks([])
ax.set_yticks([])

ax = axes[1,0]

hex_exp.plot(ax=ax,column='U_per',cmap=cmap1,norm=norm1,alpha=1)
counties_coast.plot(ax=ax,facecolor='none',edgecolor='k',alpha=1,lw=county_lw)
hydro.plot(ax=ax,color='k',lw=hydro_lw)
ax.set_ylabel('Uninsured damage',weight='bold',fontsize=ft+2)


ax.set_xticks([])
ax.set_yticks([])

ax = axes[1,1]

hex_exp.plot(ax=ax,column='U_M',cmap=cmap2,norm=norm2,alpha=1)
counties_coast.plot(ax=ax,facecolor='none',edgecolor='k',alpha=1,lw=county_lw)
hydro.plot(ax=ax,color='k',lw=hydro_lw)

ax.set_xticks([])
ax.set_yticks([])

ax = axes[2,0]

hex_exp.plot(ax=ax,column='R_per',cmap=cmap1,norm=norm1,alpha=1,legend=False)
counties_coast.plot(ax=ax,facecolor='none',edgecolor='k',alpha=1,lw=county_lw)
hydro.plot(ax=ax,color='k',lw=hydro_lw)
ax.set_ylabel('Repetitive damage',weight='bold',fontsize=ft+2)

ax.set_xticks([])
ax.set_yticks([])

ax = axes[2,1]

hex_exp.plot(ax=ax,column='R_M',cmap=cmap2,norm=norm2,alpha=1)
counties_coast.plot(ax=ax,facecolor='none',edgecolor='k',alpha=1,lw=county_lw)
hydro.plot(ax=ax,color='k',lw=hydro_lw)

ax.set_xticks([])
ax.set_yticks([])

ax = axes[2,0]
info = ax.collections[0]
axins=fig.add_axes([-0.0075,-0.03,0.575,0.25])

formatter = mpl.ticker.LogFormatter(10, labelOnlyBase=False) 
cbar = plt.colorbar(info, ax=axins, extend='both',orientation='horizontal',shrink=0.7,label='Annualized flood damage\nper structure, USD',format=formatter)
axins.axis('off')

ax = axes[2,1]
info = ax.collections[0]
axins=fig.add_axes([0.455,-0.03,0.575,0.25])
formatter = mpl.ticker.LogFormatter(10, labelOnlyBase=False) 
cbar = plt.colorbar(info, ax=axins, extend='both',orientation='horizontal',shrink=0.7,label='Cumulative flood damage over\nthe study period, millions USD',format=formatter)
axins.axis('off')


# Add lettering
letters = ['(a)','(b)','(c)','(d)','(e)','(f)']

for i,ax in enumerate(axes.flat):
    ax.text(0.025, 0.9, letters[i], transform=ax.transAxes, size=13, weight='bold')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_5_damage_hexagons.png')
fig.savefig(outname,dpi=600,bbox_inches='tight')

fig.show()


### FIGURE S10-S17 PRE-PROCESSING*** 

exp_by_event_CAMA.set_index(['year','CAMA'],inplace=True)
exp_by_event_urban.set_index(['year','urban'],inplace=True)
exp_by_event_SFHA.set_index(['year','SFHA'],inplace=True)

hex_exp = exposure_df[exposure_df['total_real_cost'] > 0]

hex_exp = pd.merge(hex_exp,structure_info[['building_id','tile_id']],on='building_id',how='left')
hex_exp = hex_exp[['year','tile_id','flood_damage']].groupby(['year','tile_id']).sum().reset_index()
hex_exp = pd.merge(hexagons[['tile_id','bldg_count','geometry']],hex_exp,on='tile_id',how='left').dropna()

hex_exp['flood_damage_prob'] = hex_exp['flood_damage']/hex_exp['bldg_count']
hex_exp['flood_damage_prevalence'] = hex_exp['flood_damage_prob']*1000

cmap1 = plt.get_cmap('GnBu')
norm1=mpl.colors.LogNorm(vmin=1, vmax=100)

county_lw = 0.85
hydro_lw = 0.25


### FIGURE S10 *** 

event_name = 'Fran'
year = 1996

normalize=1e6
unit_label = 'millions'

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(8,4))

## Part I: COST BY COMPARATIVE GROUP
ax = axes[0]

order = [0,1]
UNR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR_urban = exp_by_event_urban.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_urban = exp_by_event_urban.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_urban = exp_by_event_urban.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_urban = exp_by_event_urban.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR = np.concatenate((UNR_urban,UNR_CAMA,UNR_SFHA))
UR = np.concatenate((UR_urban,UR_CAMA,UR_SFHA))
IR = np.concatenate((IR_urban,IR_CAMA,IR_SFHA))
INR = np.concatenate((INR_urban,INR_CAMA,INR_SFHA))


U = UNR + UR
I = INR + IR
T = U+I
R = UR + IR

U_left = np.zeros(len(UNR))
I_left = U
R_left = UNR

yticks = [0,1,3,4,6,7]
yticklabels = ['Rural','Urban','Inland','Coastal','Non-SFHA','SFHA']

bottom = np.zeros(len(xticks))

# Insured
ax.barh(yticks,I,left=I_left,color=I_color,alpha=alpha,ec='k',lw=1,label='Insured')

# Uninsured
ax.barh(yticks,U,left=U_left,color=U_color,alpha=alpha,ec='k',lw=1,label='Uninsured')

# Repetitive
ax.barh(yticks,R,left=R_left,facecolor='none',hatch=hatch,ec='k',lw=1,label='Repetitive')

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.legend()

ax.set_xlabel(f'Flood damage exposure, {unit_label} USD')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.175),ncol=3,fontsize=8)

## Part II: MAP
ax = axes[1]

hex_exp[hex_exp['year']==year].plot(ax=ax,column='flood_damage_prevalence',cmap=cmap1,norm=norm1,alpha=1)
counties_coast.plot(ax=ax,facecolor='none',edgecolor='k',alpha=1,lw=county_lw)
hydro.plot(ax=ax,color='k',lw=hydro_lw)

ax.set_xticks([])
ax.set_yticks([])

minx, miny, maxx, maxy = counties_coast.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.axis('off')

info = ax.collections[0]
axins=fig.add_axes([0.29,0.2,1,0.225])
formatter = mpl.ticker.LogFormatter(10, labelOnlyBase=False) 
cbar = plt.colorbar(info,ax=axins,extend='both',orientation='horizontal',shrink=0.7,label='Flood damage prevalence\nper 1000 structures',format=formatter)
axins.axis('off')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S10_Fran.png')
fig.savefig(outname,dpi=400)

fig.show()


### FIGURE S11 *** 

event_name = 'Bonnie'
year = 1998

normalize=1e6
unit_label = 'millions'

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(8,4))

## Part I: COST BY COMPARATIVE GROUP
ax = axes[0]

order = [0,1]
UNR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR_urban = exp_by_event_urban.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_urban = exp_by_event_urban.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_urban = exp_by_event_urban.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_urban = exp_by_event_urban.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR = np.concatenate((UNR_urban,UNR_CAMA,UNR_SFHA))
UR = np.concatenate((UR_urban,UR_CAMA,UR_SFHA))
IR = np.concatenate((IR_urban,IR_CAMA,IR_SFHA))
INR = np.concatenate((INR_urban,INR_CAMA,INR_SFHA))


U = UNR + UR
I = INR + IR
T = U+I
R = UR + IR

U_left = np.zeros(len(UNR))
I_left = U
R_left = UNR

yticks = [0,1,3,4,6,7]
yticklabels = ['Rural','Urban','Inland','Coastal','Non-SFHA','SFHA']

bottom = np.zeros(len(xticks))

# Insured
ax.barh(yticks,I,left=I_left,color=I_color,alpha=alpha,ec='k',lw=1,label='Insured')

# Uninsured
ax.barh(yticks,U,left=U_left,color=U_color,alpha=alpha,ec='k',lw=1,label='Uninsured')

# Repetitive
ax.barh(yticks,R,left=R_left,facecolor='none',hatch=hatch,ec='k',lw=1,label='Repetitive')

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.legend()

ax.set_xlabel(f'Flood damage exposure, {unit_label} USD')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.175),ncol=3,fontsize=8)

## Part II: MAP
ax = axes[1]

hex_exp[hex_exp['year']==year].plot(ax=ax,column='flood_damage_prevalence',cmap=cmap1,norm=norm1,alpha=1)
counties_coast.plot(ax=ax,facecolor='none',edgecolor='k',alpha=1,lw=county_lw)
hydro.plot(ax=ax,color='k',lw=hydro_lw)

ax.set_xticks([])
ax.set_yticks([])

minx, miny, maxx, maxy = counties_coast.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.axis('off')

info = ax.collections[0]
axins=fig.add_axes([0.29,0.2,1,0.225])
formatter = mpl.ticker.LogFormatter(10, labelOnlyBase=False) 
cbar = plt.colorbar(info,ax=axins,extend='both',orientation='horizontal',shrink=0.7,label='Flood damage prevalence\nper 1000 structures',format=formatter)
axins.axis('off')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S11_Bonnie.png')
fig.savefig(outname,dpi=400)

fig.show()



### FIGURE S12 *** 

event_name = 'Floyd'
year = 1999

normalize=1e6
unit_label = 'millions'

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(8,4))

## Part I: COST BY COMPARATIVE GROUP
ax = axes[0]

order = [0,1]
UNR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR_urban = exp_by_event_urban.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_urban = exp_by_event_urban.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_urban = exp_by_event_urban.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_urban = exp_by_event_urban.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR = np.concatenate((UNR_urban,UNR_CAMA,UNR_SFHA))
UR = np.concatenate((UR_urban,UR_CAMA,UR_SFHA))
IR = np.concatenate((IR_urban,IR_CAMA,IR_SFHA))
INR = np.concatenate((INR_urban,INR_CAMA,INR_SFHA))


U = UNR + UR
I = INR + IR
T = U+I
R = UR + IR

U_left = np.zeros(len(UNR))
I_left = U
R_left = UNR

yticks = [0,1,3,4,6,7]
yticklabels = ['Rural','Urban','Inland','Coastal','Non-SFHA','SFHA']

bottom = np.zeros(len(xticks))

# Insured
ax.barh(yticks,I,left=I_left,color=I_color,alpha=alpha,ec='k',lw=1,label='Insured')

# Uninsured
ax.barh(yticks,U,left=U_left,color=U_color,alpha=alpha,ec='k',lw=1,label='Uninsured')

# Repetitive
ax.barh(yticks,R,left=R_left,facecolor='none',hatch=hatch,ec='k',lw=1,label='Repetitive')

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.legend()

ax.set_xlabel(f'Flood damage exposure, {unit_label} USD')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.175),ncol=3,fontsize=8)

## Part II: MAP
ax = axes[1]

hex_exp[hex_exp['year']==year].plot(ax=ax,column='flood_damage_prevalence',cmap=cmap1,norm=norm1,alpha=1)
counties_coast.plot(ax=ax,facecolor='none',edgecolor='k',alpha=1,lw=county_lw)
hydro.plot(ax=ax,color='k',lw=hydro_lw)

ax.set_xticks([])
ax.set_yticks([])

minx, miny, maxx, maxy = counties_coast.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.axis('off')

info = ax.collections[0]
axins=fig.add_axes([0.29,0.2,1,0.225])
formatter = mpl.ticker.LogFormatter(10, labelOnlyBase=False) 
cbar = plt.colorbar(info,ax=axins,extend='both',orientation='horizontal',shrink=0.7,label='Flood damage prevalence\nper 1000 structures',format=formatter)
axins.axis('off')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S12_Floyd.png')
fig.savefig(outname,dpi=400)

fig.show()


### FIGURE S13 *** 

event_name = 'Isabel'
year = 2003

normalize=1e6
unit_label = 'millions'

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(8,4))

## Part I: COST BY COMPARATIVE GROUP
ax = axes[0]

order = [0,1]
UNR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR_urban = exp_by_event_urban.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_urban = exp_by_event_urban.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_urban = exp_by_event_urban.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_urban = exp_by_event_urban.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR = np.concatenate((UNR_urban,UNR_CAMA,UNR_SFHA))
UR = np.concatenate((UR_urban,UR_CAMA,UR_SFHA))
IR = np.concatenate((IR_urban,IR_CAMA,IR_SFHA))
INR = np.concatenate((INR_urban,INR_CAMA,INR_SFHA))


U = UNR + UR
I = INR + IR
T = U+I
R = UR + IR

U_left = np.zeros(len(UNR))
I_left = U
R_left = UNR

yticks = [0,1,3,4,6,7]
yticklabels = ['Rural','Urban','Inland','Coastal','Non-SFHA','SFHA']

bottom = np.zeros(len(xticks))

# Insured
ax.barh(yticks,I,left=I_left,color=I_color,alpha=alpha,ec='k',lw=1,label='Insured')

# Uninsured
ax.barh(yticks,U,left=U_left,color=U_color,alpha=alpha,ec='k',lw=1,label='Uninsured')

# Repetitive
ax.barh(yticks,R,left=R_left,facecolor='none',hatch=hatch,ec='k',lw=1,label='Repetitive')

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.legend()

ax.set_xlabel(f'Flood damage exposure, {unit_label} USD')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.175),ncol=3,fontsize=8)

## Part II: MAP
ax = axes[1]

hex_exp[hex_exp['year']==year].plot(ax=ax,column='flood_damage_prevalence',cmap=cmap1,norm=norm1,alpha=1)
counties_coast.plot(ax=ax,facecolor='none',edgecolor='k',alpha=1,lw=county_lw)
hydro.plot(ax=ax,color='k',lw=hydro_lw)

ax.set_xticks([])
ax.set_yticks([])

minx, miny, maxx, maxy = counties_coast.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.axis('off')

info = ax.collections[0]
axins=fig.add_axes([0.29,0.2,1,0.225])
formatter = mpl.ticker.LogFormatter(10, labelOnlyBase=False) 
cbar = plt.colorbar(info,ax=axins,extend='both',orientation='horizontal',shrink=0.7,label='Flood damage prevalence\nper 1000 structures',format=formatter)
axins.axis('off')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S13_Isabel.png')
fig.savefig(outname,dpi=400)

fig.show()


### FIGURE S14 *** 

event_name = 'Irene'
year = 2011

normalize=1e6
unit_label = 'millions'

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(8,4))

## Part I: COST BY COMPARATIVE GROUP
ax = axes[0]

order = [0,1]
UNR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR_urban = exp_by_event_urban.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_urban = exp_by_event_urban.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_urban = exp_by_event_urban.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_urban = exp_by_event_urban.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR = np.concatenate((UNR_urban,UNR_CAMA,UNR_SFHA))
UR = np.concatenate((UR_urban,UR_CAMA,UR_SFHA))
IR = np.concatenate((IR_urban,IR_CAMA,IR_SFHA))
INR = np.concatenate((INR_urban,INR_CAMA,INR_SFHA))


U = UNR + UR
I = INR + IR
T = U+I
R = UR + IR

U_left = np.zeros(len(UNR))
I_left = U
R_left = UNR

yticks = [0,1,3,4,6,7]
yticklabels = ['Rural','Urban','Inland','Coastal','Non-SFHA','SFHA']

bottom = np.zeros(len(xticks))

# Insured
ax.barh(yticks,I,left=I_left,color=I_color,alpha=alpha,ec='k',lw=1,label='Insured')

# Uninsured
ax.barh(yticks,U,left=U_left,color=U_color,alpha=alpha,ec='k',lw=1,label='Uninsured')

# Repetitive
ax.barh(yticks,R,left=R_left,facecolor='none',hatch=hatch,ec='k',lw=1,label='Repetitive')

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.legend()

ax.set_xlabel(f'Flood damage exposure, {unit_label} USD')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.175),ncol=3,fontsize=8)

## Part II: MAP
ax = axes[1]

hex_exp[hex_exp['year']==year].plot(ax=ax,column='flood_damage_prevalence',cmap=cmap1,norm=norm1,alpha=1)
counties_coast.plot(ax=ax,facecolor='none',edgecolor='k',alpha=1,lw=county_lw)
hydro.plot(ax=ax,color='k',lw=hydro_lw)

ax.set_xticks([])
ax.set_yticks([])

minx, miny, maxx, maxy = counties_coast.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.axis('off')

info = ax.collections[0]
axins=fig.add_axes([0.29,0.2,1,0.225])
formatter = mpl.ticker.LogFormatter(10, labelOnlyBase=False) 
cbar = plt.colorbar(info,ax=axins,extend='both',orientation='horizontal',shrink=0.7,label='Flood damage prevalence\nper 1000 structures',format=formatter)
axins.axis('off')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S14_Irene.png')
fig.savefig(outname,dpi=400)

fig.show()


### FIGURE S15 *** 

event_name = 'Matthew'
year = 2016

normalize=1e6
unit_label = 'millions'

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(8,4))

## Part I: COST BY COMPARATIVE GROUP
ax = axes[0]

order = [0,1]
UNR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR_urban = exp_by_event_urban.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_urban = exp_by_event_urban.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_urban = exp_by_event_urban.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_urban = exp_by_event_urban.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR = np.concatenate((UNR_urban,UNR_CAMA,UNR_SFHA))
UR = np.concatenate((UR_urban,UR_CAMA,UR_SFHA))
IR = np.concatenate((IR_urban,IR_CAMA,IR_SFHA))
INR = np.concatenate((INR_urban,INR_CAMA,INR_SFHA))


U = UNR + UR
I = INR + IR
T = U+I
R = UR + IR

U_left = np.zeros(len(UNR))
I_left = U
R_left = UNR

yticks = [0,1,3,4,6,7]
yticklabels = ['Rural','Urban','Inland','Coastal','Non-SFHA','SFHA']

bottom = np.zeros(len(xticks))

# Insured
ax.barh(yticks,I,left=I_left,color=I_color,alpha=alpha,ec='k',lw=1,label='Insured')

# Uninsured
ax.barh(yticks,U,left=U_left,color=U_color,alpha=alpha,ec='k',lw=1,label='Uninsured')

# Repetitive
ax.barh(yticks,R,left=R_left,facecolor='none',hatch=hatch,ec='k',lw=1,label='Repetitive')

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.legend()

ax.set_xlabel(f'Flood damage exposure, {unit_label} USD')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.175),ncol=3,fontsize=8)

## Part II: MAP
ax = axes[1]

hex_exp[hex_exp['year']==year].plot(ax=ax,column='flood_damage_prevalence',cmap=cmap1,norm=norm1,alpha=1)
counties_coast.plot(ax=ax,facecolor='none',edgecolor='k',alpha=1,lw=county_lw)
hydro.plot(ax=ax,color='k',lw=hydro_lw)

ax.set_xticks([])
ax.set_yticks([])

minx, miny, maxx, maxy = counties_coast.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.axis('off')

info = ax.collections[0]
axins=fig.add_axes([0.29,0.2,1,0.225])
formatter = mpl.ticker.LogFormatter(10, labelOnlyBase=False) 
cbar = plt.colorbar(info,ax=axins,extend='both',orientation='horizontal',shrink=0.7,label='Flood damage prevalence\nper 1000 structures',format=formatter)
axins.axis('off')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S15_Matthew.png')
fig.savefig(outname,dpi=400)

fig.show()


### FIGURE S16 *** 

event_name = 'Florence'
year = 2018

normalize=1e6
unit_label = 'millions'

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(8,4))

## Part I: COST BY COMPARATIVE GROUP
ax = axes[0]

order = [0,1]
UNR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_CAMA = exp_by_event_CAMA.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR_urban = exp_by_event_urban.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_urban = exp_by_event_urban.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_urban = exp_by_event_urban.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_urban = exp_by_event_urban.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'UNR'].to_numpy()/normalize
UR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'UR'].to_numpy()/normalize
IR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'IR'].to_numpy()/normalize
INR_SFHA = exp_by_event_SFHA.loc[year].loc[order,'INR'].to_numpy()/normalize

UNR = np.concatenate((UNR_urban,UNR_CAMA,UNR_SFHA))
UR = np.concatenate((UR_urban,UR_CAMA,UR_SFHA))
IR = np.concatenate((IR_urban,IR_CAMA,IR_SFHA))
INR = np.concatenate((INR_urban,INR_CAMA,INR_SFHA))


U = UNR + UR
I = INR + IR
T = U+I
R = UR + IR

U_left = np.zeros(len(UNR))
I_left = U
R_left = UNR

yticks = [0,1,3,4,6,7]
yticklabels = ['Rural','Urban','Inland','Coastal','Non-SFHA','SFHA']

bottom = np.zeros(len(xticks))

# Insured
ax.barh(yticks,I,left=I_left,color=I_color,alpha=alpha,ec='k',lw=1,label='Insured')

# Uninsured
ax.barh(yticks,U,left=U_left,color=U_color,alpha=alpha,ec='k',lw=1,label='Uninsured')

# Repetitive
ax.barh(yticks,R,left=R_left,facecolor='none',hatch=hatch,ec='k',lw=1,label='Repetitive')

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.legend()

ax.set_xlabel(f'Flood damage exposure, {unit_label} USD')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.175),ncol=3,fontsize=8)

## Part II: MAP
ax = axes[1]

hex_exp[hex_exp['year']==year].plot(ax=ax,column='flood_damage_prevalence',cmap=cmap1,norm=norm1,alpha=1)
counties_coast.plot(ax=ax,facecolor='none',edgecolor='k',alpha=1,lw=county_lw)
hydro.plot(ax=ax,color='k',lw=hydro_lw)

ax.set_xticks([])
ax.set_yticks([])

minx, miny, maxx, maxy = counties_coast.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.axis('off')

info = ax.collections[0]
axins=fig.add_axes([0.29,0.2,1,0.225])
formatter = mpl.ticker.LogFormatter(10, labelOnlyBase=False) 
cbar = plt.colorbar(info,ax=axins,extend='both',orientation='horizontal',shrink=0.7,label='Flood damage prevalence\nper 1000 structures',format=formatter)
axins.axis('off')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S16_Florence.png')
fig.savefig(outname,dpi=400)

fig.show()


### *** FIGURE 7 PRE-PROCESSING *** ###

## Read in data from borrower simulation model

mortgage_sim_dir = '/proj/characklab/flooddata/NC/multiple_events/analysis/mortgage_borrower_simulation_base_case_postprocessed'

damaged_sim_filepath = os.path.join(mortgage_sim_dir,'simulation_output_damaged.parquet')
damaged_sim_df = pd.read_parquet(damaged_sim_filepath)

# Read in data on inflation
inflation = pd.read_csv('/proj/characklab/flooddata/NC/multiple_events/financial_data/inflation_measures.csv')
inflation['year'] = pd.to_datetime(inflation['DATE']).dt.year
inflation=inflation.rename(columns={'USACPIALLMINMEI':'cpi'})[['year','cpi']].groupby('year').mean()
cpi_by_year = inflation['cpi']
reference_year = 2020
reference_cpi = cpi_by_year[reference_year]

# Add inflation multiplier info to dataframe of borrowers with damage
damaged_sim_df['year'] = damaged_sim_df['date'].dt.year
damaged_sim_df['cpi'] = damaged_sim_df['year'].apply(lambda x: cpi_by_year[x])
damaged_sim_df['inflation_multiplier'] = reference_cpi/damaged_sim_df['cpi']

borrower_counts_filepath = os.path.join(mortgage_sim_dir,'borrower_counts.parquet')
borrower_counts = pd.read_parquet(borrower_counts_filepath)
borrower_counts = borrower_counts[borrower_counts['county'].isin(study_area_counties)]

# Create income categories
income_quantiles = pd.read_csv(os.path.join(mortgage_sim_dir,'income_quantiles.csv'))
income_quantiles['period'] = pd.to_datetime(income_quantiles['period']).dt.to_period('M')
income_quantiles = income_quantiles.set_index('period')
income_quantiles['P0'] = 0.0
income_quantiles['P100'] = np.inf
income_quantiles = income_quantiles[['P0','P20','P40','P60','P80','P100']]

income_binning_function = lambda x: np.digitize(x['monthly_income'],bins=income_quantiles.loc[x['period']].values)

damaged_sim_df['income_group'] = damaged_sim_df.apply(income_binning_function,axis=1)
damaged_sim_df['income_group'].value_counts(normalize=True)

# Create property value categories
pv_quantiles = pd.read_csv(os.path.join(mortgage_sim_dir,'property_value_quantiles.csv'))
pv_quantiles['period'] = pd.to_datetime(pv_quantiles['period']).dt.to_period('M')
pv_quantiles = pv_quantiles.set_index('period')
pv_quantiles['P0'] = 0.0
pv_quantiles['P100'] = np.inf
pv_quantiles = pv_quantiles[['P0','P20','P40','P60','P80','P100']]

pv_binning_function = lambda x: np.digitize(x['property_value'],bins=pv_quantiles.loc[x['period']].values)

damaged_sim_df['property_value_group'] = damaged_sim_df.apply(pv_binning_function,axis=1)
damaged_sim_df['property_value_group'].value_counts(normalize=True)

# Create loan age categories
loan_age_bins = np.array([0,2,5,10,np.inf])*12
damaged_sim_df['loan_age_group'] = pd.cut(damaged_sim_df['loan_age'],loan_age_bins)

# Get data on uninsured borrowers with flood damage
# Exclude pre-1999 storms (Fran, Bonnie) since the mortgage data only seems reliable post-1995
uninsured_borrower_df = damaged_sim_df[damaged_sim_df['uninsured_damage'] > 0]
uninsured_borrower_df['year'] = uninsured_borrower_df['period'].dt.year
uninsured_borrower_df = uninsured_borrower_df[uninsured_borrower_df['year'] >= 1999]

def characterize_default_risk(df,stratification_columns=['income_group'],LTV_threshold=1.0,DTI_threshold=0.45):
    
    df['strategic'] = (df['aLTV'] > LTV_threshold).astype(int)
    df['cashflow'] = (df['aDTI'] > DTI_threshold).astype(int)
    df['double_trigger'] = df['strategic']*df['cashflow']
    df['strategic_only'] = ((df['strategic']==1)&(df['double_trigger']==0)).astype(int)
    df['cashflow_only'] = ((df['cashflow']==1)&(df['double_trigger']==0)).astype(int)
    
    df = df[stratification_columns+['replicate','strategic_only','double_trigger','cashflow_only']].groupby(stratification_columns+['replicate']).sum().reset_index().drop(columns='replicate').groupby(stratification_columns).mean()
    
    return(df)

risk_by_income_group = characterize_default_risk(uninsured_borrower_df,stratification_columns=['income_group'])
risk_by_pv_group = characterize_default_risk(uninsured_borrower_df,stratification_columns=['property_value_group'])
risk_by_loan_age_group = characterize_default_risk(uninsured_borrower_df,stratification_columns=['loan_age_group'])
risk_by_county = characterize_default_risk(uninsured_borrower_df,stratification_columns=['county'])



### *** FIGURE 6 *** ###

# Look at distributions of LTV & DTI before and after storm

max_DTI = 1.2
max_LTV = 3.0

# Overall

DTI1 = uninsured_borrower_df['DTI'].to_numpy()
aDTI1 = uninsured_borrower_df['aDTI'].to_numpy()

LTV1 = uninsured_borrower_df['LTV'].to_numpy()
aLTV1 = uninsured_borrower_df['aLTV'].to_numpy()

DTI1_kde = stats.gaussian_kde(DTI1)
DTI1_pdf = np.vectorize(DTI1_kde.pdf)
aDTI1_kde = stats.gaussian_kde(aDTI1)
aDTI1_pdf = np.vectorize(aDTI1_kde.pdf)

LTV1_kde = stats.gaussian_kde(LTV1)
LTV1_pdf = np.vectorize(LTV1_kde.pdf)
aLTV1_kde = stats.gaussian_kde(aLTV1)
aLTV1_pdf = np.vectorize(aLTV1_kde.pdf)

# Lowest

DTI2 = uninsured_borrower_df[uninsured_borrower_df['income_group']==1]['DTI'].to_numpy()
aDTI2 = uninsured_borrower_df[uninsured_borrower_df['income_group']==1]['aDTI'].to_numpy()

LTV2 = uninsured_borrower_df[uninsured_borrower_df['property_value_group']==1]['LTV'].to_numpy()
aLTV2 = uninsured_borrower_df[uninsured_borrower_df['property_value_group']==1]['aLTV'].to_numpy()

DTI2_kde = stats.gaussian_kde(DTI2)
DTI2_pdf = np.vectorize(DTI2_kde.pdf)
aDTI2_kde = stats.gaussian_kde(aDTI2)
aDTI2_pdf = np.vectorize(aDTI2_kde.pdf)

LTV2_kde = stats.gaussian_kde(LTV2)
LTV2_pdf = np.vectorize(LTV2_kde.pdf)
aLTV2_kde = stats.gaussian_kde(aLTV2)
aLTV2_pdf = np.vectorize(aLTV2_kde.pdf)

# Highest

DTI3 = uninsured_borrower_df[uninsured_borrower_df['income_group']==5]['DTI'].to_numpy()
aDTI3 = uninsured_borrower_df[uninsured_borrower_df['income_group']==5]['aDTI'].to_numpy()

LTV3 = uninsured_borrower_df[uninsured_borrower_df['property_value_group']==5]['LTV'].to_numpy()
aLTV3 = uninsured_borrower_df[uninsured_borrower_df['property_value_group']==5]['aLTV'].to_numpy()

DTI3_kde = stats.gaussian_kde(DTI3)
DTI3_pdf = np.vectorize(DTI3_kde.pdf)
aDTI3_kde = stats.gaussian_kde(aDTI3)
aDTI3_pdf = np.vectorize(aDTI3_kde.pdf)

LTV3_kde = stats.gaussian_kde(LTV3)
LTV3_pdf = np.vectorize(LTV3_kde.pdf)
aLTV3_kde = stats.gaussian_kde(aLTV3)
aLTV3_pdf = np.vectorize(aLTV3_kde.pdf)

alpha=0.4

fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(8,9))

ax = axes[0,0]
x = np.linspace(0,max_DTI,250)

ax.plot(x,DTI1_pdf(x),color='C0',lw=2,label='Pre-flood (DTI)')
ax.fill_between(x,DTI1_pdf(x),color='C0',alpha=alpha)

ax.plot(x,aDTI1_pdf(x),color='C3',lw=2,label='Post-flood (ADTI)')
ax.fill_between(x,aDTI1_pdf(x),color='C3',alpha=alpha)

ax.axvline(x=0.45,c='k',ls='--',label='Threshold ADTI')

ax.set_xlim([0,max_DTI])
ax.set_ylim([0,None])
ax.set_xlabel('Debt-to-income ratio\n(Overall)')
ax.set_ylabel('Probability density')

ax.legend()

ax = axes[0,1]
x = np.linspace(0,max_LTV,250)

ax.plot(x,LTV1_pdf(x),color='C0',lw=2,label='Pre-flood (CLTV)')
ax.fill_between(x,LTV1_pdf(x),color='C0',alpha=alpha)

ax.plot(x,aLTV1_pdf(x),color='C3',lw=2,label='Post-flood (ACLTV)')
ax.fill_between(x,aLTV1_pdf(x),color='C3',alpha=alpha)

ax.axvline(x=1.0,c='k',ls='--',label='Threshold ACLTV')

ax.set_xlim([0,max_LTV])
ax.set_ylim([0,None])
ax.set_xlabel('Combined loan-to-value ratio\n(Overall)')
ax.set_ylabel('Probability density')

ax.legend()

# Lowest

ax = axes[1,0]
x = np.linspace(0,max_DTI,250)

ax.plot(x,DTI2_pdf(x),color='C0',lw=2,label='Pre-flood (DTI)')
ax.fill_between(x,DTI2_pdf(x),color='C0',alpha=alpha)

ax.plot(x,aDTI2_pdf(x),color='C3',lw=2,label='Post-flood (ADTI)')
ax.fill_between(x,aDTI2_pdf(x),color='C3',alpha=alpha)

ax.axvline(x=0.45,c='k',ls='--',label='Threshold aDTI')

ax.set_xlim([0,max_DTI])
ax.set_ylim([0,None])
#ax.set_title('All Borrowers',weight='bold')
ax.set_xlabel('Debt-to-income ratio\n(Bottom income quintile)')
ax.set_ylabel('Probability density')

ax = axes[1,1]
x = np.linspace(0,max_LTV,250)

ax.plot(x,LTV2_pdf(x),color='C0',lw=2,label='Pre-flood (CLTV)')
ax.fill_between(x,LTV2_pdf(x),color='C0',alpha=alpha)

ax.plot(x,aLTV2_pdf(x),color='C3',lw=2,label='Post-flood (ACLTV)')
ax.fill_between(x,aLTV2_pdf(x),color='C3',alpha=alpha)

ax.axvline(x=1.0,c='k',ls='--',label='Threshold ACLTV')

ax.set_xlim([0,max_LTV])
ax.set_ylim([0,None])
ax.set_xlabel('Combined loan-to-value ratio\n(Bottom property value quintile)')
ax.set_ylabel('Probability density')


# Highest

ax = axes[2,0]
x = np.linspace(0,max_DTI,250)

ax.plot(x,DTI3_pdf(x),color='C0',lw=2,label='Pre-flood (DTI)')
ax.fill_between(x,DTI3_pdf(x),color='C0',alpha=alpha)

ax.plot(x,aDTI3_pdf(x),color='C3',lw=2,label='Post-flood (ADTI)')
ax.fill_between(x,aDTI3_pdf(x),color='C3',alpha=alpha)

ax.axvline(x=0.45,c='k',ls='--',label='Threshold ADTI')

ax.set_xlim([0,max_DTI])
ax.set_ylim([0,None])
ax.set_xlabel('Debt-to-income ratio\n(Top income quintile)')
ax.set_ylabel('Probability density')

ax = axes[2,1]
x = np.linspace(0,max_LTV,250)

ax.plot(x,LTV3_pdf(x),color='C0',lw=2,label='Pre-flood (CLTV)')
ax.fill_between(x,LTV3_pdf(x),color='C0',alpha=alpha)

ax.plot(x,aLTV3_pdf(x),color='C3',lw=2,label='Post-flood (ACLTV)')
ax.fill_between(x,aLTV3_pdf(x),color='C3',alpha=alpha)

ax.axvline(x=1.0,c='k',ls='--',label='Threshold ACLTV')

ax.set_xlim([0,max_LTV])
ax.set_ylim([0,None])
ax.set_xlabel('Combined loan-to-value ratio\n(Top property value quintile)')
ax.set_ylabel('Probability density')

# Add lettering
letters = ['(a)','(b)','(c)','(d)','(e)','(f)']

for i,ax in enumerate(axes.flat):
    ax.text(-0.1, 1.1, letters[i], transform=ax.transAxes, size=13, weight='bold')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_6_LTV_DTI_distributions.png')
fig.savefig(outname,dpi=400)

fig.show()


### *** FIGURE 7 PRE-PROCESSING *** ###
unpaid_loan_balance = uninsured_borrower_df['unpaid_balance_on_all_loans'].to_numpy()
property_value = uninsured_borrower_df['property_value'].to_numpy()

monthly_income = uninsured_borrower_df['monthly_income'].to_numpy()
monthly_obligations = uninsured_borrower_df['monthly_debt_obligations'].to_numpy()

monthly_interest_rate = (uninsured_borrower_df['market_rate'].to_numpy()/100)/12

loan_term = 360
CLTV_threshold=1.0
DTI_threshold=0.45
    
B_liquidity_constraint = ((1-(1+monthly_interest_rate)**(-1*loan_term))/monthly_interest_rate)*(monthly_income*DTI_threshold-monthly_obligations)
B_equity_constraint = property_value*CLTV_threshold - unpaid_loan_balance

B_equity_constraint = np.maximum(B_equity_constraint,0)
B_liquidity_constraint = np.maximum(B_liquidity_constraint,0)

borrowing_capacity = np.minimum(B_equity_constraint,B_liquidity_constraint)

uninsured_borrower_df['nominal_borrowing_capacity'] = borrowing_capacity
uninsured_borrower_df['nominal_shortfall'] = uninsured_borrower_df['uninsured_damage'] - uninsured_borrower_df['nominal_borrowing_capacity']

# Adjust shortfall for inflation
uninsured_borrower_df['real_shortfall'] = uninsured_borrower_df['nominal_shortfall']*uninsured_borrower_df['inflation_multiplier']
uninsured_borrower_df['real_damage_cost'] = uninsured_borrower_df['uninsured_damage']*uninsured_borrower_df['inflation_multiplier']
default_df = uninsured_borrower_df[uninsured_borrower_df['real_shortfall'] > 0].reset_index(drop=True)


### *** FIGURE 7 PART I *** ###

columns = ['strategic_only','double_trigger','cashflow_only']
column_labels = ['Strategic only','Double-trigger','Cashflow only']
colors = ['C3','C4','C0']

n_replicates = 10
w=1/n_replicates
increment = 5e3
xmin=0
xmax=120e3
bins = np.arange(xmin,xmax+increment,increment)
xpos = bins[:-1]+increment/2

x1 = default_df[default_df['strategic_only']==1]['real_shortfall'].to_numpy()
x2 = default_df[default_df['double_trigger']==1]['real_shortfall'].to_numpy()
x3 = default_df[default_df['cashflow_only']==1]['real_shortfall'].to_numpy()
#x_norm = uninsured_borrower_df['real_shortfall'].to_numpy()

h1,b1 = np.histogram(x1,bins=bins,weights=w*np.ones(x1.shape))
h2,b2 = np.histogram(x2,bins=bins,weights=w*np.ones(x2.shape))
h3,b3 = np.histogram(x3,bins=bins,weights=w*np.ones(x3.shape))
#h_norm,b_norm = np.histogram(x_norm,bins=bins,weights=w*np.ones(x_norm.shape))

#h1 = h1/h_norm*100
#h2 = h2/h_norm*100
#h3 = h3/h_norm*100

fig,ax = plt.subplots(figsize=(6,4))

ax.bar(xpos,h1,width=increment,bottom=0,color=colors[0],alpha=0.6,ec='k',lw=1,label=column_labels[0])
ax.bar(xpos,h2,width=increment,bottom=h1,color=colors[1],alpha=0.6,ec='k',lw=1,label=column_labels[1])
ax.bar(xpos,h3,width=increment,bottom=h1+h2,color=colors[2],alpha=0.6,ec='k',lw=1,label=column_labels[2])

ax.set_xlim([bins.min(),bins.max()])
ax.set_ylim([0,700])

ax.xaxis.set_minor_locator(ticker.MultipleLocator(increment))
ax.xaxis.set_major_locator(ticker.MultipleLocator(4*increment))

ax.tick_params(axis='both', which='major',width=1,length=6)
ax.tick_params(axis='both', which='minor',width=1,length=3)

ax.set_xlabel('Shortfall in funding for home repairs, USD')
ax.set_ylabel('Number of mortgage borrowers\nat risk of flood-related default')

ax.legend()

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_7_part_I_shortfall_distribution.png')
fig.savefig(outname,dpi=400)

fig.show()


print(default_df['real_damage_cost'].quantile([0.25,0.5,0.75]).round(-2))
print('')
print(default_df['real_shortfall'].quantile([0.25,0.5,0.75]).round(-2))

print(np.round(100*np.mean(default_df['real_shortfall'] > 42500),2))
print(np.round(100*np.mean(default_df['real_shortfall'] > 10000),2))

print('\nstrategic_only')
print(default_df[default_df['strategic_only']==1]['real_shortfall'].quantile([0.25,0.5,0.75]).round(-2))
print(default_df[default_df['strategic_only']==1]['real_shortfall'].mean().round(-2))

print('\ncashflow_only')
print(default_df[default_df['cashflow_only']==1]['real_shortfall'].quantile([0.25,0.5,0.75]).round(-2))
print(default_df[default_df['cashflow_only']==1]['real_shortfall'].mean().round(-2))

print('\ndouble_trigger')
print(default_df[default_df['double_trigger']==1]['real_shortfall'].quantile([0.25,0.5,0.75]).round(-2))
print(default_df[default_df['double_trigger']==1]['real_shortfall'].mean().round(-2))


# Numbers we needed for Section 3.2 text

m = (uninsured_borrower_df['income_group']==1)
print(100*uninsured_borrower_df[m]['cashflow'].mean())

m = (default_df['income_group']==1)
print(100*default_df[m]['double_trigger'].mean())

m = (default_df['income_group']==5)
print(100*default_df[m]['cashflow'].mean())

m = (uninsured_borrower_df['property_value_group']==1)
print(100*uninsured_borrower_df[m]['strategic'].mean())

m = (uninsured_borrower_df['property_value_group']==5)
print(100*uninsured_borrower_df[m]['strategic'].mean())

print(100*(default_df['property_value_group'] >= 4).mean())

m = (uninsured_borrower_df['property_value_group']==5)
print(100*uninsured_borrower_df[m]['cashflow'].mean())

loan_age_groups = uninsured_borrower_df['loan_age_group'].sort_values().unique()

m = (default_df['loan_age_group']==loan_age_groups[0])
print(100*np.mean(m))

m1 = (uninsured_borrower_df['loan_age_group']==loan_age_groups[0])
m2 = (uninsured_borrower_df['loan_age_group']==loan_age_groups[3])

r1 = np.mean((uninsured_borrower_df[m1]['aDTI'] > 0.45)|(uninsured_borrower_df[m1]['aLTV'] > 1.0))
r2 = np.mean((uninsured_borrower_df[m2]['aDTI'] > 0.45)|(uninsured_borrower_df[m2]['aLTV'] > 1.0))

print(r1/r2)



### *** FIGURE 7 PART II *** ###

wedgeprops={'alpha':0.6,'edgecolor':'k','linewidth':2.0}

x = [default_df['strategic_only'].sum(),default_df['cashflow_only'].sum(),default_df['double_trigger'].sum()]

startangle=360*np.sum(x[1:3])/np.sum(x)*0.62

fig,ax = plt.subplots(figsize=(3,3))

wedges = ax.pie(x,colors=['C3','C0','C4'],startangle=startangle,wedgeprops=wedgeprops)

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_7_part_II_default_type_pie.png')
fig.savefig(outname,dpi=400)

fig.show()

counts = default_df[['strategic_only','double_trigger','cashflow_only']].sum()/n_replicates
total = counts.sum()
prop = counts/total

print('total:',np.round(total).astype(int))
print(np.round(100*prop,1))

# Compare against number of originations:
originations_path = os.path.join(pwd,'2024-07-19_distributions/hmda_mortgage_originations.csv')
originations = pd.read_csv(originations_path,index_col=0,dtype={'county_code':str,'census_tract':str})
originations = originations.rename(columns={'census_tract':'censusTract','county_code':'countyCode','census_year':'censusYear'})
originations = pd.merge(originations,counties[['countyCode','countyName']],how='left',on='countyCode')
originations = originations[originations['countyName'].isin(study_area_counties)]
originations = originations[(originations['year'] >= 1996)&(originations['year'] <= 2019)]

num_originations = len(originations)

print(' ')
print('num_originations:',num_originations)
print('percent_at_risk:',np.round(100*total/num_originations,2))



### *** FIGURE 8 *** ###

xmax=3000

columns = ['strategic_only','double_trigger','cashflow_only']
column_labels = ['Strategic only','Double-trigger','Cashflow only']
colors = ['C3','C4','C0']

fig,axes = plt.subplots(nrows=3,ncols=1,figsize=(8,8))

ax = axes[0]

yticklabels = ['Bottom','Lower\nMiddle','Middle','Upper\nMiddle','Top']
yticks = np.arange(len(yticklabels))
left = np.zeros(len(yticks))

for i in range(0,len(column_labels)):
    width = risk_by_income_group.loc[:,columns[i]].to_numpy()
    ax.barh(yticks,width,label=column_labels[i],left=left,color=colors[i],alpha=0.6,ec='k',lw=1)
    left += width
    
ax.set_xlim([0,xmax])
    
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.set_xlabel('Number of mortgage borrowers at risk of flood-related default')
ax.set_ylabel('Income quintile',fontsize=12)

ax.legend()

ax = axes[1]

yticklabels = ['Bottom','Lower\nMiddle','Middle','Upper\nMiddle','Top']
yticks = np.arange(len(yticklabels))
left = np.zeros(len(yticks))

for i in range(0,len(column_labels)):
    width = risk_by_pv_group.loc[:,columns[i]].to_numpy()
    ax.barh(yticks,width,label=column_labels[i],left=left,color=colors[i],alpha=0.6,ec='k',lw=1)
    left += width

ax.set_xlim([0,xmax])
    
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.set_xlabel('Number of mortgage borrowers at risk of flood-related default')
ax.set_ylabel('Property value quintile',fontsize=12)

ax = axes[2]

yticklabels = ['<2 years','2-4 years','5-9 years','≥10 years']
yticks = np.arange(len(yticklabels))
left = np.zeros(len(yticks))

for i in range(0,len(column_labels)):
    width = risk_by_loan_age_group.loc[:,columns[i]].to_numpy()
    ax.barh(yticks,width,label=column_labels[i],left=left,color=colors[i],alpha=0.6,ec='k',lw=1)
    left += width
    
ax.set_xlim([0,xmax])
    
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.set_xlabel('Number of mortgage borrowers at risk of flood-related default')
ax.set_ylabel('Loan age',fontsize=12)

# Add lettering
letters = ['(a)','(b)','(c)']

for i,ax in enumerate(axes.flat):
    ax.text(-0.05, 1.1, letters[i], transform=ax.transAxes, size=13, weight='bold')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_8_default_risk_by_group.png')
fig.savefig(outname,dpi=400)

fig.show()



# Get overlap in buildings flooded between each event
years = exposure_df['year'].unique()

y1_list = []
y2_list = [] 
n_overlap_list = []
p_overlap_list = []

for i,y1 in enumerate(years):
    for j,y2 in enumerate(years):
        
        df1 = exposure_df[exposure_df['year']==y1]
        df2 = exposure_df[exposure_df['year']==y2]
        
        b1 = df1['building_id'].unique()
        b2 = df2['building_id'].unique()
        
        n_overlap = np.isin(b1,b2).sum()
        p_overlap = np.round(100*n_overlap/len(np.unique(np.concatenate([b1,b2]))),1)
        
        y1_list.append(y1)
        y2_list.append(y2)
        n_overlap_list.append(n_overlap)
        p_overlap_list.append(p_overlap)
        
df = pd.DataFrame(data={'y1':y1_list,'y2':y2_list,'n_overlap':n_overlap_list,'p_overlap':p_overlap_list})
df.sort_values(by='p_overlap',ascending=False)



### *** FIGURE S17-S18 PRE-PROCESSING *** ###

relative_damage_df = exposure_df.copy()
relative_damage_df['quarter'] = relative_damage_df['period'].dt.to_timestamp().dt.to_period('Q').astype(str)

kriged_dir = '/proj/characklab/flooddata/NC/multiple_events/analysis/property_value_estimates_by_county'
kriged_counties = np.sort([x for x in os.listdir(kriged_dir) if os.path.isdir(os.path.join(kriged_dir,x)) and x != '.ipynb_checkpoints'])

rd_included_quarters = relative_damage_df['quarter'].unique()
rd_included_counties = relative_damage_df['countyName'].unique()
rd_included_building_ids = relative_damage_df['building_id'].unique()

df_list = []

for county in kriged_counties:
    
    if county in rd_included_counties:
        
        print(county)
    
        filepath = os.path.join(kriged_dir,county,f'{county}_property_values_kriged.parquet')
        df = pq.read_table(filepath,columns=['building_id','period','val_transfer_kriged'],use_pandas_metadata=True).to_pandas().rename(columns={'period':'quarter'})
        df = df[(df['quarter'].isin(rd_included_quarters))&(df['building_id'].isin(rd_included_building_ids))].rename(columns={'val_transfer_kriged':'property_value'})
        df_list.append(df)
        
df = pd.concat(df_list)
relative_damage_df = pd.merge(relative_damage_df,df,on=['building_id','quarter'],how='left')
relative_damage_df = relative_damage_df[~relative_damage_df['property_value'].isna()]

relative_damage_df['percent_damage'] = 100*relative_damage_df['nominal_cost']/relative_damage_df['property_value']

# Bin by property value qunitile
relative_damage_df['property_value_group'] = relative_damage_df.apply(pv_binning_function,axis=1)

# Get share of damaged properties in each property value quintile for each event
exp_by_event_pv_group = relative_damage_df[['year','property_value_group','building_id']].groupby(by=['year','property_value_group']).count().reset_index()
exp_by_event_pv_group = exp_by_event_pv_group.pivot(index='year',columns='property_value_group')
exp_by_event_pv_group.columns = 1+np.arange(5)

for year in years:
    exp_by_event_pv_group.loc[year] = 100*exp_by_event_pv_group.loc[year]/exp_by_event_pv_group.loc[year].sum()
    
outname = os.path.join(outfolder,'exp_by_event_pv_group.csv')
exp_by_event_pv_group.to_csv(outname)



### *** FIGRURE S17 *** ###

M = exp_by_event_pv_group.to_numpy().T

years = [1996,1998,1999,2003,2011,2016,2018]
events = ['Fran','Bonnie','Floyd','Isabel','Irene','Matthew','Florence']
event_labels = [f'{event}\n({year})' for event,year in zip(events,years)]

fig,ax = plt.subplots(figsize=(8,4))

im = ax.imshow(M,origin='lower',aspect='auto',cmap='Blues',vmin=0,vmax=80)

ax.set_xticks(np.arange(7))
ax.set_xticklabels(event_labels)

ax.set_yticks(np.arange(5))

ylabels = ['Bottom','Lower\nMiddle','Middle','Upper\nMiddle','Top']
ax.set_yticklabels(ylabels)

ax.set_ylabel('Property value quintile')

fig.colorbar(im,orientation='vertical',label='Share of flood-damaged properties, %')

for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        
        val = int(np.round(M[i,j]))
        
        if val > 40:
            color='w'
        else:
            color='k'
            
        ax.text(j,i,f'{val}%',color=color,ha='center',va='center')
        
fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S17_pv_group_share_of_damaged_properties_by_event.png')
fig.savefig(outname)

fig.show()




### *** FIGURE S18 *** ###

quantiles = [0.25,0.5,0.75]

fig,axes = plt.subplots(nrows=4,ncols=2,figsize=(8,9))

labels = ['Top','Upper middle','Middle','Lower middle','Bottom']
colors = ['#90d743', '#35b779', '#21908d', '#31688e', '#443983']

years = [1996,1998,1999,2003,2011,2016,2018]
events = ['Fran','Bonnie','Floyd','Isabel','Irene','Matthew','Florence']
event_labels = [f'{event} ({year})' for event,year in zip(events,years)]

for i,event in enumerate(events):
    
    ai = i // 2
    aj = i - 2*ai
    
    ax = axes[ai,aj]
    
    print(event)
    
    lines = []

    for j,pv_group in enumerate([5,4,3,2,1]):

        m = (relative_damage_df['property_value_group']==pv_group)&(relative_damage_df['event']==event)
        d1 = dm.empirical_distribution(relative_damage_df[m]['percent_damage'].to_numpy())
        qq = np.round(d1.ppf(quantiles),1)

        x = np.linspace(0,100,101)
        y = 100*d1.cdf(x)

        L = ax.plot(x,y,label=labels[j],color=colors[j],lw=2)
        lines.append(L.copy())

        percent_half_totaled = np.round(100*(1-d1.cdf(50)),1)
        percent_totaled = np.round(100*(1-d1.cdf(100)),1)

        print(pv_group,qq,percent_half_totaled,percent_totaled)

    ax.set_xticks(np.arange(0,100+1,25))
    ax.set_yticks(np.arange(0,100+1,25))

    ax.grid('on')

    ax.set_xlim([0,100])
    ax.set_ylim([0,100])

    ax.set_xlabel('Relative damage threshold, %')
    ax.set_ylabel('Proportion of damaged\nhomes below threshold, %')
    
    ax.set_title(event_labels[i],fontweight='bold')
    
lines,labels = ax.get_legend_handles_labels()

ax = axes[-1,-1]
ax.legend(lines, labels,title='Property value quintile',loc='center')
ax.axis('off')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S18_relative_damage_by_event.png')
fig.savefig(outname,dpi=400)

fig.show()




m1 = relative_damage_df['property_value_group'].isin([1])
m2 = relative_damage_df['year'].isin([1999,2016,2018])

(relative_damage_df[m1&m2]['percent_damage'] > 90).mean()*100




### Get characteristics of borrowers exposed to flood damage

agg_func = lambda x: len(np.unique(x))

num_loans_exposed_to_flood_damage = damaged_sim_df[['replicate','loan_id']].groupby('replicate').agg(agg_func).mean().values[0]

exposed_SFHA_building_ids = exposure_df[exposure_df['SFHA']==1]['building_id'].unique()
m = (damaged_sim_df['building_id'].isin(exposed_SFHA_building_ids))
num_SFHA_loans_exposed_to_flood_damage = damaged_sim_df[m][['replicate','loan_id']].groupby('replicate').agg(agg_func).mean().values[0]

m = (damaged_sim_df['uninsured_damage'] > 0)
num_loans_exposed_to_uninsured_damage = damaged_sim_df[m][['replicate','loan_id']].groupby('replicate').agg(agg_func).mean().values[0]

# Inflation adjust damages
damaged_sim_df['real_insured_damage'] = damaged_sim_df['insured_damage']*damaged_sim_df['inflation_multiplier']
damaged_sim_df['real_uninsured_damage'] = damaged_sim_df['uninsured_damage']*damaged_sim_df['inflation_multiplier']

print(damaged_sim_df[m]['real_uninsured_damage'].quantile([0.25,0.5,0.75]).round(-2))

damaged_sim_df['relative_uninsured_damage'] = damaged_sim_df['uninsured_damage']/damaged_sim_df['property_value']
print((100*damaged_sim_df[m]['relative_uninsured_damage'].quantile([0.25,0.5,0.75])).round())

delta_DTI = (100*(damaged_sim_df[m]['aDTI'] - damaged_sim_df[m]['DTI'])).median().round(1)
delta_LTV = (100*(damaged_sim_df[m]['aLTV'] - damaged_sim_df[m]['LTV'])).median().round(1)

print(delta_LTV,delta_DTI)


m_inc = (damaged_sim_df['income_group']==1)
m_pv = (damaged_sim_df['property_value_group']==1)

delta_DTI_low = (100*(damaged_sim_df[m&m_inc]['aDTI'] - damaged_sim_df[m&m_inc]['DTI'])).median().round(1)
delta_LTV_low = (100*(damaged_sim_df[m&m_pv]['aLTV'] - damaged_sim_df[m&m_pv]['LTV'])).median().round(1)

print(delta_LTV_low,delta_DTI_low)


percent_aLTV_over_threshold = (100*(damaged_sim_df[m]['aLTV'] > 1.0).mean()).round()
percent_aDTI_over_threshold = (100*(damaged_sim_df[m]['aDTI'] > 0.45).mean()).round()
percent_both_over_threshold = (100*((damaged_sim_df[m]['aDTI'] > 0.45)&(damaged_sim_df[m]['aLTV'] > 1.0)).mean()).round()

print(percent_aLTV_over_threshold,percent_aDTI_over_threshold,percent_both_over_threshold)


### *** FIGURE S19: MULTI-WAY SENSITIVITY ANALYSIS *** ###

sensitivity_filepath = os.path.join(pwd,'sensitivity_analysis_data/sensitivity_analysis_data.csv')
sens_df = pd.read_csv(sensitivity_filepath)

sens_df = sens_df.set_index(['dc_mult','pv_mult','rr_mult'])
reference_area = np.ceil(sens_df['total'].max()/1000)*1000
reference_radius = 0.35

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(9.5,4.5))
ft=12

wedgeprops={'alpha':0.6,'edgecolor':'k','linewidth':0.5}
pad=0.4

for i,rr_mult in enumerate([1.0,0.5]):
    
    ax = axes[i]
    
    for j,pv_mult in enumerate([0.8,1.0,1.2]):
        for k,dc_mult in enumerate([0.8,1.0,1.2]):
            
            row = sens_df.loc[dc_mult,pv_mult,rr_mult]
            
            x = [row['strategic_only'],row['cashflow_only'],row['double_trigger']]
            startangle=360*np.sum(x[1:3])/np.sum(x)*0.62
            
            total = int(row.to_numpy()[-1])

            radius = np.sqrt(total/reference_area)*reference_radius
            
            total_str = '{:,}'.format(total) 
            
            ax.pie(x,colors=['C3','C0','C4'],startangle=90,wedgeprops=wedgeprops,radius=radius,center=(k,j),frame=True)
            ax.text(k,j-pad,f'N={total_str}',ha='center',va='center',fontsize=ft-3)
                    
    ax.set_xlim([-0.5,2.5])
    ax.set_ylim([-0.5,2.5])
    ax.set_xticks([0,1,2])
    ax.set_yticks([0,1,2])
    
    ax.axvline(x=0.5,color='k',lw=0.5)
    ax.axvline(x=1.5,color='k',lw=0.5)
    ax.axhline(y=0.5,color='k',lw=0.5)
    ax.axhline(y=1.5,color='k',lw=0.5)
    
    ax.set_xticklabels(['-20%','0%','+20%'])
    ax.set_yticklabels(['-20%','0%','+20%'])
    
    ax.set_xlabel('Change in flood damage costs\nrelative to baseline estimate')
    ax.set_ylabel('Change in property values\nrelative to baseline estimate')
    ax.tick_params(axis='both', length=0) 

axes[0].set_title('Market interest rates',fontweight='bold')
axes[1].set_title('50% of market interest rates',fontweight='bold')

# Add lettering
letters = ['(a)','(b)','(c)','(d)','(e)','(f)']

for i,ax in enumerate(axes.flat):
    ax.text(-0.15, 1.05, letters[i], transform=ax.transAxes, size=13, weight='bold')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_S19_sensitivity_analysis.png')
fig.savefig(outname,dpi=400)

fig.show()


xmax=3000

columns = ['strategic_only','double_trigger','cashflow_only']
column_labels = ['Strategic only','Double-trigger','Cashflow only']
colors = ['C3','C4','C0']

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,4))

yticklabels = ['Bottom','Lower\nMiddle','Middle','Upper\nMiddle','Top']
yticks = np.arange(len(yticklabels))
left = np.zeros(len(yticks))

for i in range(0,len(column_labels)):
    width = risk_by_income_group.loc[:,columns[i]].to_numpy()
    ax.barh(yticks,width,label=column_labels[i],left=left,color=colors[i],alpha=0.6,ec='k',lw=1)
    left += width
    
ax.set_xlim([0,xmax])
    
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.set_xlabel('Number of mortgage borrowers at risk of flood-related default')
ax.set_ylabel('Income quintile',fontsize=12)

ax.legend(ncol=3,bbox_to_anchor=(1, 1.15), loc='upper right',columnspacing=1,handletextpad=0.5)
ax.axis('off')

fig.tight_layout()
outname = os.path.join(outfolder,'Figure_S19_legend.png')
fig.savefig(outname,dpi=400)

fig.show()


### *** FIGURE 9: GRANT SENSITIVITY ANALYSIS *** ###

columns = ['strategic_only','double_trigger','cashflow_only']
column_labels = ['Strategic only','Double-trigger','Cashflow only']
colors = ['C3','C4','C0']

n_replicates = len(default_df['replicate'].unique())

# grant_amounts = [0,10e3,20e3,30e3,42.5e3,50e3,60e3]
# grant_labels = ['None\n(base case)','$10k','$20k','$30k','$42.5k\n(IHP max)','$50k','$60k']

grant_amounts = np.array([0,5,10,15,20,25,30,35,42.5])*1e3
grant_labels = ['None\n(base case)','$5k','$10k','$15k','$20k','$25k','$30k','$35k','$42.5k\n(IHP max)']

NS_arr = np.zeros(len(grant_amounts))
ND_arr = np.zeros(len(grant_amounts))
NC_arr = np.zeros(len(grant_amounts))

for i,grant in enumerate(grant_amounts):
    
    m = (default_df['real_shortfall'] > grant)

    NS = default_df[m]['strategic_only'].sum()/n_replicates
    ND = default_df[m]['double_trigger'].sum()/n_replicates
    NC = default_df[m]['cashflow_only'].sum()/n_replicates
    
    NS_arr[i] = NS
    ND_arr[i] = ND
    NC_arr[i] = NC
    
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,4))

xpos = np.arange(len(grant_amounts))

width=0.67

ax.bar(xpos,NS_arr,width=width,bottom=0,color=colors[0],alpha=0.6,ec='k',lw=1,label=column_labels[0])
ax.bar(xpos,ND_arr,width=width,bottom=NS_arr,color=colors[1],alpha=0.6,ec='k',lw=1,label=column_labels[1])
ax.bar(xpos,NC_arr,width=width,bottom=NS_arr+ND_arr,color=colors[2],alpha=0.6,ec='k',lw=1,label=column_labels[2])

ax.set_xticks(xpos)
ax.set_xticklabels(grant_labels)

ax.legend()

ax.set_xlabel('Home repair grant amount, USD')
ax.set_ylabel('Number of mortgage borrowers\nat risk of flood-related default')

fig.tight_layout()

outname = os.path.join(outfolder,'Figure_9_grant_sensitivity_analysis.png')
fig.savefig(outname,dpi=400)

fig.show()


total_arr = NS_arr+ND_arr+NC_arr
reduction_arr = 100*(total_arr - total_arr[0])/total_arr[0]
for i in range(len(reduction_arr)):
    print(grant_amounts[i],np.round(reduction_arr[i]))




