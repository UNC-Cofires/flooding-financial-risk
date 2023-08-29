import os
import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime as dt
import floodprediction as fp

### *** LOAD MAIN DATA SOURCES *** ###

# Specify current working directory
pwd = os.getcwd()

# Get event-specific information
eventlist_path = os.path.join(pwd,'flood_event_list.csv')
eventlist = pd.read_csv(eventlist_path)
event_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
event = eventlist.loc[event_idx].to_dict()
event_date = pd.to_datetime(event['start_date'])

# Specify path to input file geodatabase
input_gdb_path = '/proj/characklab/flooddata/NC/data_joining/2023-02-27_joined_data/included_data.gdb'

parcels = gpd.read_file(input_gdb_path,layer='parcels')
print('Finished reading parcels',flush=True)
buildings = gpd.read_file(input_gdb_path,layer='buildings')
print('Finished reading buildings',flush=True)
claims = gpd.read_file(input_gdb_path,layer='claims')
print('Finished reading claims',flush=True)
policies = gpd.read_file(input_gdb_path,layer='policies')
print('Finished reading policies',flush=True)

# Create column representing amount paid out in claims
claims['total_payout'] = claims['Net_Total_Payments']

# Attach environmental data sampled at building points
raster_values_path = '/proj/characklab/flooddata/NC/multiple_events/data_processing/rasters/raster_values_at_building_points.csv'
raster_values = pd.read_csv(raster_values_path)
buildings = pd.merge(buildings,raster_values,on='building_id',how='left')

# Rename SFHA columns (this will make it easier to match up with OpenFEMA later)
buildings = buildings.rename(columns={'NC_SFHA_NoX_extend_10252022':'SFHA'})

# Specify coordinate reference system to use in analysis (should be projected rather than geographic)
crs = claims.crs

# Read in data on NC counties
counties_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/NC_counties'
counties = gpd.read_file(counties_path).to_crs(crs)
counties = counties[['FIPS','geometry']].rename(columns={'FIPS':'countyCode'})

# Read in data on NC census tracts
# (Use those from 2010 census to match up with OpenFEMA)
census_tracts_path = f'/proj/characklab/flooddata/NC/multiple_events/geospatial_data/TIGER/nc_2010_census_tracts_clean'
census_tracts = gpd.read_file(census_tracts_path).to_crs(crs)
census_tracts = census_tracts.rename(columns={'GEOID':'censusTract'})

# Read in data on NC HUC6 watersheds
watersheds_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/USGS/NC_HUC6'
watersheds = gpd.read_file(watersheds_path).to_crs(crs)
watersheds = watersheds[['huc6','geometry']]

### *** PRE-PROCESS MAIN DATA SOURCES ***

# Attach county codes to buildings
buildings = gpd.sjoin(buildings,counties,how='left',predicate='within').drop(columns='index_right')

# Attach census tract codes to buildings
# (use sjoin_nearest instead of sjoin since some coastal buildings fall just outside tract boundaries)
buildings = gpd.sjoin_nearest(buildings,census_tracts,how='left',max_distance=1000).drop(columns='index_right')

# Attach HUC6 watershed codes to buildings
buildings = gpd.sjoin(buildings,watersheds,how='left',predicate='within').drop(columns='index_right')

# Impute any missing values from nearest non-missing neighbor
buildings = fp.impute_missing_spatially(buildings)

# Convert occupancy type and foundation type codes to braod categories
lookup_table_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/NC Emergency Management Spatial Data/buildings_lookup_tables.xlsx'
occup_type_lookup_table = pd.read_excel(lookup_table_path,sheet_name='OCCUP_TYPE')
occup_type_lookup_table = occup_type_lookup_table[['Code','Label']].set_index('Code')
found_type_lookup_table = pd.read_excel(lookup_table_path,sheet_name='FOUND_TYPE')
found_type_lookup_table = found_type_lookup_table[['Code','Label']].set_index('Code')

def clean_occup_type(x):
    try:
        res = occup_type_lookup_table.loc[int(x),'Label']
    except:
        res = 'other'
    return(res)

def clean_found_type(x):
    try:
        res = found_type_lookup_table.loc[int(x),'Label']
    except:
        res = 'other'
    return(res)

buildings['OCCUP_TYPE'] = buildings['OCCUP_TYPE'].apply(clean_occup_type)
buildings['FOUND_TYPE'] = buildings['FOUND_TYPE'].apply(clean_found_type)

# Apply one-hot encoding to categorical variables
buildings = buildings.join(pd.get_dummies(buildings['huc6'],prefix='huc6')).drop(columns='huc6')
buildings = buildings.join(pd.get_dummies(buildings['OCCUP_TYPE'],prefix='OCCUP_TYPE')).drop(columns='OCCUP_TYPE')
buildings = buildings.join(pd.get_dummies(buildings['FOUND_TYPE'],prefix='FOUND_TYPE')).drop(columns='FOUND_TYPE')

### *** LOAD AUXILIARY DATA SOURCES *** ###

# Read in openfema data
openfema_claims_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/OpenFEMA/NC_FemaNfipClaims.csv'
openfema_policies_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/OpenFEMA/NC_FemaNfipPolicies.csv'
openfema_claims = pd.read_csv(openfema_claims_path,index_col=0).rename(columns={'reportedZipcode':'reportedZipCode'})
openfema_policies = pd.read_csv(openfema_policies_path,index_col=0)

# Keep only required columns
openfema_keepcols_claims = ['countyCode','censusTract','floodZone','dateOfLoss','amountPaidOnBuildingClaim','amountPaidOnContentsClaim']
openfema_keepcols_policies = ['countyCode','censusTract','floodZone','policyEffectiveDate','policyTerminationDate','policyCount']

openfema_claims = openfema_claims[openfema_keepcols_claims]
openfema_policies = openfema_policies[openfema_keepcols_policies]

# Format columns
openfema_claims['countyCode'] = openfema_claims['countyCode'].astype(str).apply(lambda x: x[2:-2])
openfema_claims['censusTract'] = openfema_claims['censusTract'].astype(str).apply(lambda x: x[:-2])
openfema_claims['dateOfLoss'] = openfema_claims['dateOfLoss'].astype(np.datetime64)
openfema_claims['floodZone'] = openfema_claims['floodZone'].astype(str)

openfema_policies['countyCode'] = openfema_policies['countyCode'].astype(str).apply(lambda x: x[2:-2])
openfema_policies['censusTract'] = openfema_policies['censusTract'].astype(str).apply(lambda x: x[:-2])
openfema_policies['policyEffectiveDate'] = openfema_policies['policyEffectiveDate'].astype(np.datetime64)
openfema_policies['policyTerminationDate'] = openfema_policies['policyTerminationDate'].astype(np.datetime64)
openfema_policies['floodZone'] = openfema_policies['floodZone'].astype(str)

# Create binary variable denoting location inside/outside SFHA
openfema_claims['SFHA'] = openfema_claims['floodZone'].apply(lambda x: x.startswith('A') or x.startswith('V')).astype(int)
openfema_policies['SFHA'] = openfema_policies['floodZone'].apply(lambda x: x.startswith('A') or x.startswith('V')).astype(int)

# Format monetary amounts
openfema_claims['amountPaidOnBuildingClaim'] = openfema_claims['amountPaidOnBuildingClaim'].fillna(0)
openfema_claims['amountPaidOnContentsClaim'] = openfema_claims['amountPaidOnContentsClaim'].fillna(0)
openfema_claims['total_payout'] = openfema_claims['amountPaidOnBuildingClaim'] + openfema_claims['amountPaidOnContentsClaim']
openfema_claims = openfema_claims.dropna()
openfema_policies = openfema_policies.dropna()

### *** ADJUST DAMAGES FOR INFLATION *** ###

# Read in data on meausures of inflation
inflation_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/inflation_measures.csv'
inflation = pd.read_csv(inflation_path)
inflation['DATE'] = inflation['DATE'].astype(np.datetime64)

# Reference date used when adjusting for inflation (e.g., convert all costs to 2020 USD)
reference_date = '2020-01-01'
reference_row = inflation[inflation['DATE'] >= reference_date].iloc[0]
nominal_row = inflation[inflation['DATE'] >= event['peak_date']].iloc[0]

# Select measure of inflation to use and calculate multiplier
inflation_measure = 'USACPICORMINMEI' # CPI - All items excluding food and energy
inflation_multiplier = reference_row[inflation_measure]/nominal_row[inflation_measure]

### *** PREDICT PRESENCE / ABSENCE OF FLOOD DAMAGE *** ###

# Define study area and period
fips_list = event['county_fips'].split(',')
study_area = counties[counties['countyCode'].isin(fips_list)]

start_date = event['start_date']
end_date = event['end_date']
peak_date = event['peak_date']
event_name = event['name']

# Process address-level data.
print(event_name,flush=True)

floodevent = fp.FloodEvent(study_area,start_date,end_date,peak_date,crs)
floodevent.preprocess_data(parcels,buildings,claims,policies,inflation_multiplier)

# Use OpenFEMA data to determine where policies are missing.
# Based on this information, generate pseudo-absences
floodevent.preprocess_openfema(openfema_claims,openfema_policies,inflation_multiplier)
floodevent.stratify_missing(['censusTract','SFHA'])
floodevent.create_pseudo_absences(n_realizations=100)

# Define features used to predict presence/absence of flood damage
response_variable = 'flood_damage'

presence_absence_features = ['SFHA',
                             'NHDcoastline_DistRaster_500res_08222022',
                             'NC_MajorHydro_DistRaster_500res_08292022',
                             'HANDraster_MosaicR_IDW30_finalR_03032023',
                             'TWIrasterHuc12_10262022',
                             'soilsKsat_NC_03072023',
                             'NEDavgslope_NCcrop_huc12_10262022',
                             'NEDraster_resample_07042022',
                             'NLCDimpraster_NC2016_07132022']

huc_columns = [x for x in floodevent.training_dataset.columns if x.startswith('huc')]
presence_absence_features += huc_columns

# Remove features that are uninformative or highly correlated with others
presence_absence_features = fp.remove_unnecessary_features(presence_absence_features,floodevent.training_dataset,max_corr=0.7)

# Perform k-fold cross-validation
floodevent.cross_validate(response_variable,presence_absence_features,k=5,use_adjusted=True,threshold=0.6)

# Predict presence/absence of flooding
floodevent.predict_presence_absence(response_variable,presence_absence_features,use_adjusted=True,threshold=0.6)

### *** PREDICT COST OF DAMAGES AMONG FLOODED HOMES *** ###

# Define features used to predict cost of damage among flooded buildings
damage_cost_features = ['SFHA',
                        'YEAR_BUILT',
                        'BLDG_VALUE',
                        'HTD_SQ_FT',
                        'FFE',
                        'NHDcoastline_DistRaster_500res_08222022',
                        'NC_MajorHydro_DistRaster_500res_08292022',
                        'HANDraster_MosaicR_IDW30_finalR_03032023',
                        'TWIrasterHuc12_10262022',
                        'soilsKsat_NC_03072023',
                        'NEDavgslope_NCcrop_huc12_10262022',
                        'NEDraster_resample_07042022',
                        'NLCDimpraster_NC2016_07132022']

huc_columns = [x for x in floodevent.training_dataset.columns if x.startswith('huc')]
damage_cost_features += huc_columns

occup_columns = [x for x in floodevent.training_dataset.columns if x.startswith('OCCUP_TYPE')]
damage_cost_features += occup_columns

found_columns = [x for x in floodevent.training_dataset.columns if x.startswith('FOUND_TYPE')]
damage_cost_features += found_columns

# Remove features that are uninformative or highly correlated with others
damage_cost_features = fp.remove_unnecessary_features(damage_cost_features,floodevent.training_dataset,max_corr=0.7)

# Predict cost of flood-related damages among buildings predicted as flooded
floodevent.predict_damage_cost('flood_damage','total_payout',damage_cost_features)

### *** SAVE RESULTS *** ###
with open(f'{event_name}_FloodEvent.object','wb') as f:
    pickle.dump(floodevent,f)
    f.close()
