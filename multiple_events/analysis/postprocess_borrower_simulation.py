import os
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq

### *** HELPER FUNCTIONS *** ###

def kaplan_meier(time,event,t_max=360):
    """
    param: time: numpy array of failure / censoring times
    param: event: numpy array denoting whether the event occured (1) or didn't occur (0)
    param: t_max: maximum follow-up time to calculate survival for
    returns: S: kaplan-meier estimate of survival function at time t
    returns: tvals: timepoints of survival curve
    """
    
    tvals = np.arange(0,t_max+1)
    prob_event = np.zeros(len(tvals))

    for i,t in enumerate(tvals):
        nt = np.sum((time >= t))
        dt = np.sum(event[time == t])

        prob_event[i] = dt/nt

    S = np.cumprod(1-prob_event)
    S = S/S[0]
    
    return(S,tvals)

### *** MAIN *** ###

# Set up folders 
pwd = os.getcwd()

mortgage_sim_dir = '/proj/characklab/flooddata/NC/multiple_events/analysis/mortgage_borrower_simulation_base_case'

outfolder = mortgage_sim_dir + '_postprocessed'

if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)
    
# Read in data on NC counties
crs = 'EPSG:32617'
counties_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/NC_counties'
counties = gpd.read_file(counties_path).to_crs(crs)
counties = counties[['FIPS','County','geometry']].rename(columns={'FIPS':'countyCode','County':'countyName'})
counties = counties.sort_values(by='countyName').reset_index(drop=True)

# Get number of replicates
replicates = np.sort([int(x.split('_')[-1]) for x in os.listdir(mortgage_sim_dir)])

# Get number of complete county-replicate combinations
incomplete_counties = set()

for replicate in replicates:
    for county in counties['countyName']:
        test_path = os.path.join(mortgage_sim_dir,f'replicate_{replicate}/{county}/{county}_simulation_output_2019.parquet')
        if not os.path.exists(test_path):
            incomplete_counties.add(county)
            
print('Missing counties:',', '.join(incomplete_counties),flush=True)
            
complete_counties = counties['countyName'][~counties['countyName'].isin(incomplete_counties)].to_list()

damage_df_list = []
quantile_df_list = []
futime_df_list = []

for county in complete_counties:
    for replicate in replicates:
        
        print(f'{county}: replicate {replicate}',flush=True)
        
        sim_dir = os.path.join(mortgage_sim_dir,f'replicate_{replicate}/{county}')
        
        sim_files = np.sort([x for x in os.listdir(sim_dir) if 'simulation_output' in x])
        sim_filepaths = [os.path.join(sim_dir,file) for file in sim_files]
        
        for filepath in sim_filepaths:
            
            sim_df = pd.read_parquet(filepath)
            
            sim_columns = list(sim_df.columns)
            
            sim_df['county'] = county
            sim_df['replicate'] = replicate
            
            sim_df = sim_df[['county','replicate'] + sim_columns]
            
            # Get snapshot of borrower finances at time of flood damage exposure
            m = (sim_df['insured_damage'] + sim_df['uninsured_damage'] > 0)
            damage_df = sim_df[m]
            
            # Get data needed to calculate time-varying borrower income and property value quintiles
            quantile_df = sim_df[['period','monthly_income','property_value']]
            
            # Get data needed to construct loan survival curves
            futime_df = sim_df[['loan_id','loan_purpose','loan_term','loan_age','termination_code']].groupby('loan_id').last().reset_index()
            futime_df = futime_df.drop(columns='loan_id').rename(columns={'loan_age':'time','termination_code':'event'})
            futime_df['event'] = (~futime_df['event'].isna()).astype(int)
            
            # Append to list
            damage_df_list.append(damage_df)
            quantile_df_list.append(quantile_df)
            futime_df_list.append(futime_df)
    
    gc.collect()
    
# Concatenate county-level data
damage_df = pd.concat(damage_df_list).reset_index(drop=True)
quantile_df = pd.concat(quantile_df_list).reset_index(drop=True)
futime_df = pd.concat(futime_df_list).reset_index(drop=True)

# Save detailed results for flood-damaged properties
damaged_sim_outname = os.path.join(outfolder,'simulation_output_damaged.parquet')
damage_df.to_parquet(damaged_sim_outname)

# Get income and property value quintiles
quantile_df = quantile_df.groupby('period').quantile(np.arange(0.2,1,0.2)).reset_index().rename(columns={'level_1':'quantile'})

income_quant_df = quantile_df.pivot(index='period',columns='quantile',values='monthly_income')
income_quant_df.columns = [f'P{int(100*x)}' for x in income_quant_df.columns]
income_quant_df.to_csv(os.path.join(outfolder,'income_quantiles.csv'))

pv_quant_df = quantile_df.pivot(index='period',columns='quantile',values='property_value')
pv_quant_df.columns = [f'P{int(100*x)}' for x in pv_quant_df.columns]
pv_quant_df.to_csv(os.path.join(outfolder,'property_value_quantiles.csv'))

# Construct Kaplan-Meier curves
p30_futime = futime_df[(futime_df['loan_purpose']=='purchase')&(futime_df['loan_term']==360)]
r30_futime = futime_df[(futime_df['loan_purpose']=='refinance')&(futime_df['loan_term']==360)]
r15_futime = futime_df[(futime_df['loan_purpose']=='refinance')&(futime_df['loan_term']==180)]

p30_surv,p30_tvals = kaplan_meier(p30_futime['time'].to_numpy(),p30_futime['event'].to_numpy())
r30_surv,r30_tvals = kaplan_meier(r30_futime['time'].to_numpy(),r30_futime['event'].to_numpy())
r15_surv,r15_tvals = kaplan_meier(r15_futime['time'].to_numpy(),r15_futime['event'].to_numpy())

surv_df = pd.DataFrame({'loan_age':p30_tvals,'p30_surv':p30_surv,'r30_surv':r30_surv,'r15_surv':r15_surv})
surv_df.to_csv(os.path.join(outfolder,'simulated_survival.csv'),index=False)