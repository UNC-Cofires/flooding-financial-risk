import os
import gc
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq

# Set up folders
mortgage_sim_dir = '/proj/characklab/flooddata/NC/multiple_events/analysis/mortgage_borrower_simulation'
outfolder = mortgage_sim_dir + '_postprocessed'

if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)
    

# Read in data on NC counties
crs = 'EPSG:32617'
counties_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/NC_counties'
counties = gpd.read_file(counties_path).to_crs(crs)
counties = counties[['FIPS','County','geometry']].rename(columns={'FIPS':'countyCode','County':'countyName'})

n_present = 0

for i,county in enumerate(np.sort(counties['countyName'])):
    
    orig_path = os.path.join(mortgage_sim_dir,county,f'{county}_originations.parquet')
    prop_path = os.path.join(mortgage_sim_dir,county,f'{county}_properties.parquet')
    sim_path = os.path.join(mortgage_sim_dir,county,f'{county}_simulation_output.parquet')
    failed_path = os.path.join(mortgage_sim_dir,county,f'{county}_failed_loans.csv')
    
    if os.path.exists(sim_path):
        
        n_present += 1

        county_orig = pq.read_table(orig_path,use_pandas_metadata=True).to_pandas()
        county_prop = pq.read_table(prop_path,use_pandas_metadata=True).to_pandas()
        county_sim = pq.read_table(sim_path,use_pandas_metadata=True).to_pandas()
        failed_loans = pd.read_csv(failed_path)
        county_failed = county_orig[county_orig['loan_id'].isin(failed_loans['loan_id'])]
                
        if n_present > 1:
            orig_df = pd.concat([orig_df,county_orig])
            prop_df = pd.concat([prop_df,county_prop])
            sim_df = pd.concat([sim_df,county_sim])
            failed_df = pd.concat([failed_df,county_failed])
            
        else:
            orig_df = county_orig
            prop_df = county_prop
            sim_df = county_sim
            failed_df = county_failed
        
        print(f'{i+1})',county,'- Present',flush=True)
        
        gc.collect()
        
    else:
        
        print(f'{i+1})',county,'- Missing',flush=True)
    
orig_df.reset_index(drop=True,inplace=True)
prop_df.reset_index(drop=True,inplace=True)
sim_df.reset_index(drop=True,inplace=True)
failed_df.reset_index(drop=True,inplace=True)

# Split off households that experience flood damage
m = ((sim_df['insured_damage'] + sim_df['uninsured_damage']) > 0)

damaged_loan_ids = sim_df[m]['loan_id'].unique()
damaged_building_ids = sim_df[m]['building_id'].unique()

damaged_sim_df = sim_df[sim_df['loan_id'].isin(damaged_loan_ids)]
damaged_prop_df = prop_df[prop_df['building_id'].isin(damaged_building_ids)]
damaged_sim_df = pd.merge(damaged_sim_df,damaged_prop_df[['building_id','countyName','censusTract_2010','SFHA']],on='building_id',how='left')

# Get income and property value quantiles for each period
income_quant_df = sim_df[['period','monthly_income']].groupby('period').quantile(np.arange(0.2,1,0.2)).reset_index().rename(columns={'level_1':'income_quantile'})
income_quant_df = income_quant_df.pivot(index='period',columns='income_quantile',values='monthly_income')
income_quant_df.columns = [f'P{int(100*x)}' for x in income_quant_df.columns]
income_quant_df.to_csv(os.path.join(outfolder,'income_quantiles.csv'))

pv_quant_df = sim_df[['period','property_value']].groupby('period').quantile(np.arange(0.2,1,0.2)).reset_index().rename(columns={'level_1':'property_value_quantile'})
pv_quant_df = pv_quant_df.pivot(index='period',columns='property_value_quantile',values='property_value')
pv_quant_df.columns = [f'P{int(100*x)}' for x in pv_quant_df.columns]
pv_quant_df.to_csv(os.path.join(outfolder,'property_value_quantiles.csv'))

# Save concatenated results
orig_outname = os.path.join(outfolder,'originations.parquet')
prop_outname = os.path.join(outfolder,'properties.parquet')
sim_outname = os.path.join(outfolder,'simulation_output.parquet')
damaged_sim_outname = os.path.join(outfolder,'simulation_output_damaged.parquet')
failed_outname = os.path.join(outfolder,'failed_loans.parquet')

orig_df.to_parquet(orig_outname)
prop_df.to_parquet(prop_outname)
sim_df.to_parquet(sim_outname)
damaged_sim_df.to_parquet(damaged_sim_outname)
failed_df.to_parquet(failed_outname)