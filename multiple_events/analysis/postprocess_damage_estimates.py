import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime as dt
import pyarrow as pa
import pyarrow.parquet as pq
import floodprediction as fp

# Specify current working directory
pwd = os.getcwd()

# Spcify path to directory containing flood event objects
damage_dir = os.path.join(pwd,'2024-03-06_damage_estimates')

# Specify output directory for processed property-value estimates
outfolder = damage_dir + '_by_county'
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)
    
# Read in geospatial data
crs = 'EPSG:32617'

# Read in data on NC counties
counties_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/NC_counties'
counties = gpd.read_file(counties_path).to_crs(crs)
counties = counties[['FIPS','County','geometry']].rename(columns={'FIPS':'countyCode','County':'countyName'})

# Read in data on 2010 NC census tracts
census_tracts_path = f'/proj/characklab/flooddata/NC/multiple_events/geospatial_data/TIGER/nc_2010_census_tracts_clean'
census_tracts = gpd.read_file(census_tracts_path).to_crs(crs)
census_tracts = census_tracts.rename(columns={'GEOID':'censusTract'})

# Read in data on single-family detached homes
input_gdb_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/property_value_inputs/nc_property_value_data_clean.gdb'
homes = gpd.read_file(input_gdb_path,layer='single_family_homes')
homes = homes[homes['OCCUP_TYPE']=='single_family_home']
homes = homes[['building_id','parcel_id','countyCode','censusTract_1990','censusTract_2000','censusTract_2010','censusTract_2020']]

# Read in damage estimates
floodevent_filenames = np.sort(os.listdir(damage_dir))
floodevent_filepaths = [os.path.join(damage_dir,x) for x in floodevent_filenames]
floodevent_names = [x.split('_')[1] for x in floodevent_filenames]
floodevent_years = [pd.Timestamp(x.split('_')[0]).year for x in floodevent_filenames]
floodevent_labels = [f'{name}\n({year})' for name,year in zip(floodevent_names,floodevent_years)]
floodevent_periods = []
floodevent_list = []

for i,filepath in enumerate(floodevent_filepaths):
    
    print(floodevent_names[i],flush=True)
    
    with open(filepath, 'rb') as f:
        floodevent = pickle.load(f)
        
    floodevent_periods.append(pd.Period(floodevent.peak_date,freq='M'))
    floodevent_list.append(floodevent)
    
# Get flood damage exposure for each event and adjust losses for inflation

# Read in data on U.S. CPI for all items
inflation_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/inflation_measures.csv'
inflation = pd.read_csv(inflation_path)
inflation['DATE'] = pd.to_datetime(inflation['DATE'])
inflation['year'] = inflation['DATE'].dt.year
cpi_by_year = inflation.groupby('year').mean()['USACPIALLMINMEI']

reference_year = 2020
reference_cpi = cpi_by_year[reference_year]

exposure_list = []

for i in range(len(floodevent_list)):
    
    floodevent = floodevent_list[i]

    agg_df,damage_df=floodevent.aggregate_flood_damage(stratification_columns=['countyCode','censusTract','SFHA'])
    damage_df = pd.merge(damage_df,counties[['countyCode','countyName']],on='countyCode',how='left').rename(columns={'flood_damage_class':'flood_damage'})
    damage_df['uninsured'] = (damage_df['insured']==0).astype(int)
    damage_df['insured_nominal_cost'] = damage_df['insured']*damage_df['total_cost']
    damage_df['uninsured_nominal_cost'] = damage_df['uninsured']*damage_df['total_cost']
    damage_df['nominal_cost'] = damage_df['total_cost']

    # Adjust for inflation
    inflation_multiplier = reference_cpi/cpi_by_year[floodevent_years[i]]

    damage_df['insured_real_cost'] = inflation_multiplier*damage_df['insured_nominal_cost']
    damage_df['uninsured_real_cost'] = inflation_multiplier*damage_df['uninsured_nominal_cost']
    damage_df['real_cost'] = inflation_multiplier*damage_df['nominal_cost']

    damage_df['period'] = floodevent_periods[i]
    damage_df['event'] = floodevent_names[i]

    damage_df = damage_df[['building_id','period','event','countyCode','countyName','censusTract','SFHA','insured','uninsured','flood_damage','nominal_cost','insured_nominal_cost','uninsured_nominal_cost','real_cost','insured_real_cost','uninsured_real_cost']]
    
    exposure_list.append(damage_df)
    
exposure_df = pd.concat(exposure_list).reset_index(drop=True)
exposure_df['single_family_detached'] = exposure_df['building_id'].isin(homes['building_id']).astype(int)

# After a structure experiences damage for the first time, mark and subsequent damages as repetitive 
exposure_df['count_times_flooded'] = exposure_df.groupby('building_id').cumcount()+1
exposure_df['repetitive'] = (exposure_df['count_times_flooded'] >= 2).astype(int)

# Save to file
outname = os.path.join(outfolder,'statewide_flood_damage_exposure.parquet')
exposure_df.to_parquet(outname)

# Create time series of flood damage exposure at single-family detached homes
# Break these up by county since we'll run the borrower simulation model in parallel 
time_periods = pd.date_range('1990-01-01','2019-12-01',freq='MS').to_period('M')

for county_name,county_code in zip(counties['countyName'],counties['countyCode']):
    
    print(county_name,flush=True)
    
    county_folder = os.path.join(outfolder,county_name)
    if not os.path.exists(county_folder):
        os.makedirs(county_folder,exist_ok=True)
    
    # Get all single-family detached homes in county 
    county_homes = homes[homes['countyCode']==county_code]
    county_homes['countyName']=county_name

    # Attach SFHA info to all single-family homes (not just those previously included in flood damage model)
    SFHA_info_path = '/proj/characklab/flooddata/NC/multiple_events/data_processing/rasters/raster_values_at_building_points.csv'
    SFHA_info = pd.read_csv(SFHA_info_path,usecols=['building_id','NC_SFHA_NoX_extend_10252022']).rename(columns={'NC_SFHA_NoX_extend_10252022':'SFHA'})
    county_homes = pd.merge(county_homes,SFHA_info,on='building_id',how='left')
    county_homes = county_homes[['building_id','parcel_id','countyCode','countyName','censusTract_1990','censusTract_2000','censusTract_2010','censusTract_2020','SFHA']]

    # Get time series of damages at each single-family home
    timeseries_list = []

    for period in time_periods:
        temp = county_homes[['building_id']].copy()
        temp['period'] = period
        timeseries_list.append(temp)

    county_timeseries = pd.concat(timeseries_list)
    county_timeseries = pd.merge(county_timeseries,exposure_df[['building_id','period','nominal_cost','insured_nominal_cost','uninsured_nominal_cost']],how='left',on=['building_id','period']).fillna(0)
    county_timeseries.sort_values(by=['building_id','period']).reset_index(drop=True)
    
    # Save results
    outname = os.path.join(county_folder,f'{county_name}_homes.parquet')
    county_homes.to_parquet(outname)
    
    outname = os.path.join(county_folder,f'{county_name}_exposure_timeseries.parquet')
    county_timeseries.to_parquet(outname)
    
