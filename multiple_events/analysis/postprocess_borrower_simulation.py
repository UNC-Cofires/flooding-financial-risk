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

def damage_df_generator(mortgage_sim_dir,complete_counties,replicates):
    """
    Generator that loops over county-level simulations that yields information on borrowers with flood damage. 
    """
    
    for county in complete_counties:
        for replicate in replicates:

            print(f'Damage - {county}: replicate {replicate}',flush=True)

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
                
                yield damage_df
                                
def quantile_df_generator(mortgage_sim_dir,complete_counties,replicates):
    """
    Generator that loops over county-level simulations that yields information on distribution of property values and income over time. 
    """
    
    for county in complete_counties:
        for replicate in replicates:

            print(f'Quantiles - {county}: replicate {replicate}',flush=True)

            sim_dir = os.path.join(mortgage_sim_dir,f'replicate_{replicate}/{county}')

            sim_files = np.sort([x for x in os.listdir(sim_dir) if 'simulation_output' in x])
            sim_filepaths = [os.path.join(sim_dir,file) for file in sim_files]

            for filepath in sim_filepaths:

                sim_df = pd.read_parquet(filepath)

                sim_columns = list(sim_df.columns)

                sim_df['county'] = county
                sim_df['replicate'] = replicate

                sim_df = sim_df[['county','replicate'] + sim_columns]
                
                # Get data needed to calculate time-varying borrower income and property value quintiles
                quantile_df = sim_df[['replicate','county','period','monthly_income','property_value']]

                yield quantile_df
                                
def futime_df_generator(mortgage_sim_dir,complete_counties,replicates):
    """
    Generator that loops over county-level simulations that yields time to mortgage repayment. 
    """
    
    for county in complete_counties:
        for replicate in replicates:

            print(f'Follow-up time - {county}: replicate {replicate}',flush=True)

            sim_dir = os.path.join(mortgage_sim_dir,f'replicate_{replicate}/{county}')

            sim_files = np.sort([x for x in os.listdir(sim_dir) if 'simulation_output' in x])
            sim_filepaths = [os.path.join(sim_dir,file) for file in sim_files]

            for filepath in sim_filepaths:

                sim_df = pd.read_parquet(filepath)

                sim_columns = list(sim_df.columns)

                sim_df['county'] = county
                sim_df['replicate'] = replicate

                sim_df = sim_df[['county','replicate'] + sim_columns]
                
                # Get data needed to construct loan survival curves
                futime_df = sim_df[['loan_id','loan_purpose','loan_term','loan_age','termination_code']].groupby('loan_id').last().reset_index()
                futime_df = futime_df.drop(columns='loan_id').rename(columns={'loan_age':'time','termination_code':'event'})
                futime_df['event'] = (~futime_df['event'].isna()).astype(int)

                yield futime_df
                
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

# Save detailed results for flood-damaged properties
damage_df = pd.concat(damage_df_generator(mortgage_sim_dir,complete_counties,replicates))
damaged_sim_outname = os.path.join(outfolder,'simulation_output_damaged.parquet')
damage_df.to_parquet(damaged_sim_outname)

del damage_df # Free up RAM
gc.collect()

# Get info on number of borrowers over time
quantile_df = pd.concat(quantile_df_generator(mortgage_sim_dir,complete_counties,replicates))
borrower_count_df = quantile_df[['replicate','county','period','property_value']].groupby(['replicate','county','period']).count().rename(columns={'property_value':'num_borrowers'}).reset_index()
borrower_count_df.to_parquet(os.path.join(outfolder,'borrower_counts.parquet'))

# Save info on time-varying income and property value quintiles
quantile_df.drop(columns=['replicate','county'],inplace=True)
quantile_df = quantile_df.groupby('period').quantile(np.arange(0.2,1,0.2)).reset_index().rename(columns={'level_1':'quantile'})

income_quant_df = quantile_df.pivot(index='period',columns='quantile',values='monthly_income')
income_quant_df.columns = [f'P{int(100*x)}' for x in income_quant_df.columns]
income_quant_df.to_csv(os.path.join(outfolder,'income_quantiles.csv'))

pv_quant_df = quantile_df.pivot(index='period',columns='quantile',values='property_value')
pv_quant_df.columns = [f'P{int(100*x)}' for x in pv_quant_df.columns]
pv_quant_df.to_csv(os.path.join(outfolder,'property_value_quantiles.csv'))

del quantile_df # Free up RAM
gc.collect()

# Construct Kaplan-Meier curves of loan survival
futime_df = pd.concat(futime_df_generator(mortgage_sim_dir,complete_counties,replicates))

p30_mask = (futime_df['loan_purpose']=='purchase')&(futime_df['loan_term']==360)
r30_mask = (futime_df['loan_purpose']=='refinance')&(futime_df['loan_term']==360)
r15_mask = (futime_df['loan_purpose']=='refinance')&(futime_df['loan_term']==180)

p30_surv,p30_tvals = kaplan_meier(futime_df[p30_mask]['time'].to_numpy(),futime_df[p30_mask]['event'].to_numpy())
r30_surv,r30_tvals = kaplan_meier(futime_df[r30_mask]['time'].to_numpy(),futime_df[r30_mask]['event'].to_numpy())
r15_surv,r15_tvals = kaplan_meier(futime_df[r15_mask]['time'].to_numpy(),futime_df[r15_mask]['event'].to_numpy())

surv_df = pd.DataFrame({'loan_age':p30_tvals,'p30_surv':p30_surv,'r30_surv':r30_surv,'r15_surv':r15_surv})
surv_df.to_csv(os.path.join(outfolder,'simulated_survival.csv'),index=False)

del futime_df # Free up RAM
gc.collect()