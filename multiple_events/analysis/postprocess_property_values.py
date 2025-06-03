import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime as dt
import pyarrow as pa
import pyarrow.parquet as pq
import spacetimekriging as stk

### This script takes the property value estimates produced by property_value_estimation.py, 
### which are spatially stratified by kriging neighborhood,
### and regroups them so that they are spatially stratified by county. 

# Specify current working directory
pwd = os.getcwd()

# Spcify path to property value estimates by kriging neighborhood
property_value_dir = os.path.join(pwd,'property_value_estimates')

# Specify output directory for processed property-value estimates
outfolder = os.path.join(pwd,'property_value_estimates_by_county')
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

# Kriging neighborhoods
kriging_neighborhoods = pd.read_csv(os.path.join(pwd,'kriging_neighborhoods.csv'))
kriging_neighborhoods['census_tracts'] = kriging_neighborhoods['census_tracts'].apply(lambda x: x.split(','))

kn_geom = []
for i in range(len(kriging_neighborhoods)):
    m = census_tracts['censusTract'].isin(kriging_neighborhoods['census_tracts'][i])
    kn_geom.append(census_tracts[m].dissolve()['geometry'].values[0])
    
kriging_neighborhoods = gpd.GeoDataFrame(kriging_neighborhoods, geometry=kn_geom, crs=crs)

## Determine which kriging neighborhoods we have complete data for

group_num_list = []
n_complete_list = []
percent_complete_list = []

for group_num in np.arange(75):
    group_path = os.path.join(property_value_dir,f'group_{group_num}')
    kriged_estimates = np.sort([x for x in os.listdir(group_path) if 'kriged' in x])
    n_complete = len(kriged_estimates)
    percent_complete = np.round(100*n_complete/120,1)
    
    group_num_list.append(group_num)
    n_complete_list.append(n_complete)
    percent_complete_list.append(percent_complete)
    
progress_df = pd.DataFrame(data={'cluster_num':group_num_list,'n_complete':n_complete_list,'percent_complete':percent_complete_list})
m = (progress_df['percent_complete'] < 100)

incomplete_df = progress_df[m]
complete_df = progress_df[~m]

missing_list = ','.join(incomplete_df['cluster_num'].astype(str).to_list())
print(f'Kriging neighborhoods with missing data: {missing_list}',flush=True)


## Determine which counties are missing data

m = kriging_neighborhoods['cluster_num'].isin(incomplete_df['cluster_num'])
missing_neighborhoods = kriging_neighborhoods[m]
complete_neighborhoods = kriging_neighborhoods[~m]

missing_county_fips = []
for i in range(len(missing_neighborhoods)):
    missing_county_fips += list(counties[counties['geometry'].intersects(missing_neighborhoods['geometry'].values[i])]['countyCode'].unique())
    
missing_county_fips = np.sort(np.unique(missing_county_fips))
m = counties['countyCode'].isin(missing_county_fips)
missing_counties = counties[m]
complete_counties = counties[~m]

fig,ax = plt.subplots(figsize=(6,4))
complete_counties.plot(ax=ax,facecolor='C0',alpha=0.5,edgecolor='k')
missing_counties.plot(ax=ax,facecolor='C3',alpha=0.5,edgecolor='k')
fig.tight_layout()
outname = os.path.join(outfolder,'complete_counties.png')
fig.savefig(outname,dpi=400)

print(f'\nMissing counties (n={len(missing_counties)}):\n',flush=True)
for county in missing_counties['countyName'].sort_values().to_list():
    print('   ',county,flush=True)
    
print(f'\n\nComplete counties (n={len(complete_counties)}):\n',flush=True)
for county in complete_counties['countyName'].sort_values().to_list():
    print('   ',county,flush=True)
    
## Regroup kriged property value estimates (and cross-validation error) by county

complete_counties = complete_counties.sort_values(by='countyName').reset_index(drop=True)
already_processed = (complete_counties['countyName'].isin(os.listdir(outfolder)))
counties_to_process = complete_counties[~already_processed]

print(f'\n{np.sum(already_processed)} / {len(complete_counties)} counties with complete data were already processed.')
print(f'Now processing the remaining {np.sum(~already_processed)}:\n')

for row in counties_to_process.to_dict(orient="records"):
    
    county_code = row['countyCode']
    county_name = row['countyName']
    county_geom = row['geometry']
    
    print(f'    {county_name}',flush=True)
    
    intersecting_neighborhoods = kriging_neighborhoods[kriging_neighborhoods['geometry'].intersects(county_geom)]['cluster_num'].to_list()
    
    county_cv_list = []
    county_pv_list = []

    for group_num in intersecting_neighborhoods:

        group_dir = os.path.join(property_value_dir,f'group_{group_num}')

        cv_path = os.path.join(group_dir,f'group_{group_num}_cross_validation.parquet')

        # Read in cross-validation data
        table = pq.read_table(cv_path,use_pandas_metadata=True)
        group_cv_df = table.to_pandas()

        # Get entries from county of interest 
        group_cv_df = group_cv_df[group_cv_df['countyCode']==county_code]

        county_cv_list.append(group_cv_df)

        # Get kriged property value estimates for entries within county at different points in time
        periods = np.sort([x.split('_')[-1].replace('.parquet','') for x in os.listdir(group_dir) if 'kriged' in x])

        usecols = ['building_id','parcel_id','countyCode','censusTract_1990','censusTract_2000','censusTract_2010','censusTract_2020','period','date','log_val_transfer_kriged','sigma_log_val_transfer_kriged','val_transfer_kriged']

        for period in periods:

            pv_path = os.path.join(group_dir,f'group_{group_num}_property_values_kriged_{period}.parquet')

            # Read in cross-validation data
            table = pq.read_table(pv_path,columns=usecols,use_pandas_metadata=True)
            group_pv_df = table.to_pandas()

            # Get entries from county of interest 
            group_pv_df = group_pv_df[group_pv_df['countyCode']==county_code]

            county_pv_list.append(group_pv_df)
            
    # Concatenate entries across various kriging neighborhoods and timepoints
    county_cv_df = pd.concat(county_cv_list).reset_index(drop=True)
    county_pv_df = pd.concat(county_pv_list).sort_values(by=['building_id','date']).reset_index(drop=True)
    
    # Save results
    county_folder = os.path.join(outfolder,county_name)
    if not os.path.exists(county_folder):
        os.makedirs(county_folder,exist_ok=True)

    cv_outname = os.path.join(county_folder,f'{county_name}_cross_validation.parquet')
    pv_outname = os.path.join(county_folder,f'{county_name}_property_values_kriged.parquet')

    county_cv_df.to_parquet(cv_outname)
    county_pv_df.to_parquet(pv_outname)