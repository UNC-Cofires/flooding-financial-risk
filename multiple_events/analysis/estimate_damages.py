import os
import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime as dt

import floodprediction as fp

# Specify current working directory
pwd = os.getcwd()

# Specify path to input file geodatabase
input_gdb_path = '/proj/characklab/flooddata/NC/data_joining/2023-02-27_joined_data/included_data.gdb'

buildings = gpd.read_file(input_gdb_path,layer='buildings')
print('Finished reading buildings',flush=True)
claims = gpd.read_file(input_gdb_path,layer='claims')
print('Finished reading claims',flush=True)
policies = gpd.read_file(input_gdb_path,layer='policies')
print('Finished reading policies',flush=True)

# Attach environmental data sampled at building points
raster_values_path = '/proj/characklab/flooddata/NC/multiple_events/data_processing/rasters/raster_values_at_building_points.csv'
raster_values = pd.read_csv(raster_values_path)
buildings = pd.merge(buildings,raster_values,on='building_id',how='left')

# Rename SFHA columns (this will make it easier to match up with OpenFEMA later)
buildings = buildings.rename(columns={'NC_SFHA_NoX_extend_10252022':'SFHA'})

# Specify crs
crs = claims.crs

# Specify path to NC counties
counties_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/NC_counties'
counties = gpd.read_file(counties_path).to_crs(crs)

# Specify path to NC census tracts
census_tracts_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/NC_2019_census_tracts'
census_tracts = gpd.read_file(census_tracts_path).to_crs(crs)

# Convert county FIPS codes and GEOIDs to integers
census_tracts['FIPS'] = census_tracts['COUNTYFP'].astype(int)
census_tracts['censusTract'] = census_tracts['GEOID'].astype(int)

# Specify path to NC watersheds
watersheds_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/USGS/NC_HUC6'
watersheds = gpd.read_file(watersheds_path).to_crs(crs)

# Specify path to event data
eventlist_path = os.path.join(pwd,'flood_event_list.csv')
eventlist = pd.read_csv(eventlist_path)

# Attach huc6 watershed names to buildings (we'll later use this as a predictor of flood damage)
buildings = gpd.sjoin(buildings,watersheds[['huc6','geometry']],how='left',predicate='within').drop(columns='index_right')

# Attach FIPS codes to buildings (we'll later use this during post-stratification)
buildings = gpd.sjoin(buildings,counties[['FIPS','geometry']],how='left',predicate='within').drop(columns='index_right')

# Read in openfema data
openfema_claims_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/OpenFEMA/NC_FemaNfipClaims.csv'
openfema_policies_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/OpenFEMA/NC_FemaNfipPolicies.csv'
openfema_claims = pd.read_csv(openfema_claims_path,index_col=0)
openfema_policies = pd.read_csv(openfema_policies_path,index_col=0)

# Keep only required columns
openfema_keepcols_claims = ['censusTract','countyCode','floodZone','dateOfLoss','amountPaidOnBuildingClaim','amountPaidOnContentsClaim']
openfema_keepcols_policies = ['censusTract','countyCode','floodZone','policyEffectiveDate','policyTerminationDate','policyCount']

openfema_claims = openfema_claims[openfema_keepcols_claims]
openfema_policies = openfema_policies[openfema_keepcols_policies]

# Format monetary amounts
openfema_claims['amountPaidOnBuildingClaim'] = openfema_claims['amountPaidOnBuildingClaim'].fillna(0)
openfema_claims['amountPaidOnContentsClaim'] = openfema_claims['amountPaidOnContentsClaim'].fillna(0)
openfema_claims['total_payout'] = openfema_claims['amountPaidOnBuildingClaim'] + openfema_claims['amountPaidOnContentsClaim']
openfema_claims = openfema_claims.dropna()
openfema_policies = openfema_policies.dropna()

# Create binary variable denoting location inside/outside SFHA
openfema_claims['SFHA'] = openfema_claims['floodZone'].apply(lambda x: x.startswith('A') or x.startswith('V')).astype(int)
openfema_policies['SFHA'] = openfema_policies['floodZone'].apply(lambda x: x.startswith('A') or x.startswith('V')).astype(int)

# Format FIPS codes
openfema_claims['FIPS'] = (openfema_claims['countyCode'] - 37000).astype(int).apply(lambda x: "{:03d}".format(x))
openfema_policies['FIPS'] = (openfema_policies['countyCode'] - 37000).astype(int).apply(lambda x: "{:03d}".format(x))

# Format census tract codes
openfema_claims['censusTract'] = openfema_claims['censusTract'].astype(int)
openfema_policies['censusTract'] = openfema_policies['censusTract'].astype(int)

# Attach census tract geometries to dataframe
openfema_claims = pd.merge(census_tracts[['censusTract','geometry']],openfema_claims,on='censusTract',how='right')
openfema_policies = pd.merge(census_tracts[['censusTract','geometry']],openfema_policies,on='censusTract',how='right')

# Get event-specific information
event_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
event = eventlist.loc[event_idx].to_dict()

# Define study area and period
fips_list = event['county_fips'].split(',')
study_area = counties[counties['FIPS'].isin(fips_list)]

start_date = event['start_date']
end_date = event['end_date']
peak_date = event['peak_date']
event_name = event['name']

# Process data
print(event_name,flush=True)

floodevent = fp.FloodEvent(study_area,start_date,end_date,peak_date,crs)
floodevent.preprocess_data(buildings,claims,policies)

# Apply post-stratification to post-2009 events
if floodevent.start_date.year > 2009:
    floodevent.preprocess_openfema(openfema_claims,openfema_policies)
    floodevent.post_stratify(['FIPS','SFHA','flood_damage'])

# Save results
with open(f'{event_name}_FloodEvent.object','wb') as f:
    pickle.dump(floodevent,f)
    f.close()
