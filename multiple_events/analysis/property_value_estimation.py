import os
import pickle
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold,GridSearchCV
import spacetimekriging as stk

### *** HELPER FUNCTIONS *** ###
def hmda_census_year(hmda_year):
    """
    For a given HMDA data activity year, returns the census year used for tract boundaries.
    
    Pre-1992: Poor match rate across all census tract years
    1992-2002: 1990 census
    2003-2011: 2000 census
    2012-2021: 2010 census
    2022-present: 2020 census
    """
    if hmda_year < 1992:
        census_year = np.nan
    elif hmda_year <= 2002:
        census_year = 1990
    elif hmda_year <= 2011:
        census_year = 2000
    elif hmda_year <= 2021:
        census_year = 2010
    else:
        census_year = 2020
    return(census_year)

### *** SET UP FOLDERS AND ENVIRONMENT *** ###

# Specify number of available CPUs for parallel processing
n_cores = int(os.environ['SLURM_NTASKS'])

# Specify current working directory
pwd = os.getcwd()
    
# Determine which group of counties to include in estimation
group_path = os.path.join(pwd,'property_value_county_groups.csv')
group_df = pd.read_csv(group_path)
group_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
group = group_df.loc[group_idx].to_dict()

# Specify output directory for model runs
pv_folder = os.path.join(pwd,'property_value_estimates')
outfolder = os.path.join(pv_folder,f'group_{group_idx}')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)
    
### *** LOAD DATA SOURCES *** ###
crs = 'EPSG:32617'

# Read in data on NC counties
counties_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/NC_counties'
counties = gpd.read_file(counties_path).to_crs(crs)
counties = counties[['FIPS','geometry']].rename(columns={'FIPS':'countyCode'})

# Define study area
included_counties = group['counties'].split(',')
study_area = counties[counties['countyCode'].isin(included_counties)].dissolve()['geometry'].values[0]

# Apply a 5km buffer so that we avoid edge effects 
buffer = 5000
buffered_study_area = study_area.buffer(buffer)
bbox = buffered_study_area.bounds

# Read in property sales data
input_gdb_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/property_value_inputs/nc_property_value_data_clean.gdb'
sales = gpd.read_file(input_gdb_path,layer='sales_data',bbox=bbox)

# Read in list of single family homes
homes = gpd.read_file(input_gdb_path,layer='single_family_homes',bbox=bbox)

# Exclude manufactured/mobile homes
sales = sales[sales['OCCUP_TYPE'] == 'single_family_home']
homes = homes[homes['OCCUP_TYPE'] == 'single_family_home']

# Clip to study area (include sales within the buffer zone as hard points to improve kriging at edges)
sales = sales[sales.intersects(buffered_study_area)]
sales['study_area'] = sales.intersects(study_area).astype(int)
homes = homes[homes.intersects(study_area)]

### *** SPECIFY TIMEPOINTS OF PROPERTY VALUE ESTIMATION *** ###

# Get dates corresponding to midpoint of each quarter between 1990 and 2020
dates = pd.date_range('1990-01-01','2019-12-31',freq='D')
quarters = dates.to_period('Q')
q_start = dates.is_quarter_start
q_end = dates.is_quarter_end
dates = pd.to_datetime((dates[q_start] + (dates[q_end] - dates[q_start])/2).date)

timepoint_list = []

for date in dates:
    temp = homes.copy()
    temp['date'] = date
    timepoint_list.append(temp)
    
property_timepoints = pd.concat(timepoint_list).reset_index(drop=True)

t0_date = pd.Timestamp('1990-01-01')

property_timepoints['year'] = property_timepoints['date'].dt.year
property_timepoints['month'] = property_timepoints['date'].dt.month
property_timepoints['time_val'] = (property_timepoints['date'] - t0_date).dt.days

sales['date_transfer'] = sales['date_transfer'].dt.tz_localize(None)
sales['year'] = sales['date_transfer'].dt.year
sales['month'] = sales['date_transfer'].dt.month
sales['time_val'] = (sales['date_transfer'] - t0_date).dt.days

### *** SPECIFY VALUES OF TIME-DEPENDENT PREDICTORS *** ###

# Read in distribution of single-family 30-year fixed loan amounts by tract and year (from HMDA)
tract_loan_dist_path = '/proj/characklab/flooddata/NC/multiple_events/analysis/2024-02-25_distributions/tractlevel_income_loan_dist_by_year.object'
state_loan_dist_path = '/proj/characklab/flooddata/NC/multiple_events/analysis/2024-02-25_distributions/statelevel_income_loan_dist_by_year.object'

with open(tract_loan_dist_path,'rb') as f:
    tract_loan_dist = pickle.load(f)
    
with open(state_loan_dist_path,'rb') as f:
    state_loan_dist = pickle.load(f)
    
# Get median and IQR loan amount by year/census tract and join to property sales data
year_list = []
census_year_list = []
census_tract_list = []
Q1_list = []
Q2_list = []
Q3_list = []

for year in np.arange(1990,2020+1):
    hmda_year = max(year,1992)
    census_year = hmda_census_year(hmda_year)
    tract_ids = np.sort(np.unique(np.concatenate([property_timepoints[f'censusTract_{census_year}'].unique(),sales[f'censusTract_{census_year}'].unique()])))
    
    # If tract/year combo is present in HMDA data, use tract-level distribution
    for census_tract in tract_ids:
        if census_tract in tract_loan_dist[hmda_year].keys():
            if tract_loan_dist[hmda_year][census_tract]['n_obs'] > 1:
                Q1,Q2,Q3 = tract_loan_dist[hmda_year][census_tract]['nominal_loan_amount'].ppf([0.25,0.5,0.75])
            else:
                Q1 = tract_loan_dist[hmda_year][census_tract]['nominal_loan_amount']
                Q2 = Q1
                Q3 = Q1
        else:
            # If tract/year combo not in HMDA data, use state-level distribution
            Q1,Q2,Q3 = state_loan_dist[hmda_year]['nominal_loan_amount'].ppf([0.25,0.5,0.75])
            
        year_list.append(year)
        census_year_list.append(census_year)
        census_tract_list.append(census_tract)
        Q1_list.append(Q1)
        Q2_list.append(Q2)
        Q3_list.append(Q3)
        
data={'year':year_list,
      'census_year':census_year_list,
      'censusTract':census_tract_list,
      'Q1':Q1_list,
      'Q2':Q2_list,
      'Q3':Q3_list}

loan_IQR_df = pd.DataFrame(data)

for census_year in loan_IQR_df['census_year'].unique():
    temp = loan_IQR_df[loan_IQR_df['census_year']==census_year]
    temp = temp.rename(columns={'censusTract':f'censusTract_{census_year}'})
    temp = temp.drop(columns='census_year')
    sales = pd.merge(sales,temp,on=['year',f'censusTract_{census_year}'],how='left')
    property_timepoints = pd.merge(property_timepoints,temp,on=['year',f'censusTract_{census_year}'],how='left')
    
Q1_cols = property_timepoints.columns[property_timepoints.columns.str.startswith('Q1')]
Q2_cols = property_timepoints.columns[property_timepoints.columns.str.startswith('Q2')]
Q3_cols = property_timepoints.columns[property_timepoints.columns.str.startswith('Q3')]

Q1_drop = Q1_cols[Q1_cols != 'Q1']
Q2_drop = Q2_cols[Q2_cols != 'Q2']
Q3_drop = Q3_cols[Q3_cols != 'Q3']
dropcols = np.concatenate([Q1_drop,Q2_drop,Q3_drop])

sales['Q1'] = sales[Q1_cols].apply(np.nanmax,axis=1)
sales['Q2'] = sales[Q2_cols].apply(np.nanmax,axis=1)
sales['Q3'] = sales[Q3_cols].apply(np.nanmax,axis=1)

property_timepoints['Q1'] = property_timepoints[Q1_cols].apply(np.nanmax,axis=1)
property_timepoints['Q2'] = property_timepoints[Q2_cols].apply(np.nanmax,axis=1)
property_timepoints['Q3'] = property_timepoints[Q3_cols].apply(np.nanmax,axis=1)

sales = sales.drop(columns=dropcols)
property_timepoints = property_timepoints.drop(columns=dropcols)

# Also attach home price index by year and county
home_price_index_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/property_value_inputs/nc_fha_hpi_by_county_and_year.csv'
home_price_index = pd.read_csv(home_price_index_path,dtype={'countyCode':str})
sales = pd.merge(sales,home_price_index[['year','countyCode','county_hpi_missing_imputed']],on=['year','countyCode'],how='left')
property_timepoints = pd.merge(property_timepoints,home_price_index[['year','countyCode','county_hpi_missing_imputed']],on=['year','countyCode'],how='left')

# Log-transform certain variables
sales[f'log_val_transfer'] = np.log(sales['val_transfer'])
log_transform_variables = ['HTD_SQ_FT','parcel_sq_ft','BLDG_VALUE','median_hh_income','Q1','Q2','Q3']

for var in log_transform_variables:
    sales[var] = np.maximum(sales[var],1)
    property_timepoints[var] = np.maximum(property_timepoints[var],1)
    sales[f'log_{var}'] = np.log(sales[var])
    property_timepoints[f'log_{var}'] = np.log(property_timepoints[var])
    
### *** RANDOM FOREST REGRESSION KRIGING CROSS VALIDATION *** ###

response_variable = 'log_val_transfer'
features = ['log_HTD_SQ_FT',
            'log_parcel_sq_ft',
            'log_BLDG_VALUE',
            'YEAR_BUILT',
            'log_median_hh_income',
            'county_hpi_missing_imputed',
            'log_Q1',
            'log_Q2',
            'log_Q3']

spatial_lags = np.arange(0,5000+1,250)
temporal_lags = np.arange(0,1800+1,60)

k=10

kf = KFold(n_splits=k,random_state=None,shuffle=True)

cv_list = []

for i,(train_indices,test_indices) in enumerate(kf.split(sales)):
    
    # Get hard points and kriging interpolation points 
    hard_points = sales.iloc[train_indices].copy()
    krig_points = sales.iloc[test_indices].copy()
    krig_points = krig_points[krig_points['study_area']==1]
    
    X_h = hard_points[features].to_numpy()
    X_k = krig_points[features].to_numpy()
    y_h = hard_points[response_variable].to_numpy()
    
    # Determine optimal hyperparameters for mean trend random forest model
    param_grid = {'n_estimators':[200],'max_depth':[7,9,11],'max_features':[0.33,0.5,0.66]}
    param_grid = {}
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=k-1, scoring='neg_mean_squared_error',n_jobs=n_cores)
    grid_search.fit(X_h,y_h)
    hyperparams = grid_search.best_params_
    
    # Fit model to hard point data
    model = RandomForestRegressor(**hyperparams).fit(X_h,y_h)
    
    # Get residuals at hard points
    z_h = y_h - model.predict(X_h)
    hard_points['residual'] = z_h
    
    # Estimate residuals at krig points via simple kriging (assume zero mean)
    Cs = stk.SphCovFun(a_bounds=(250,5000))
    Ct = stk.ExpCovFun(a=365)
    Cst = stk.ProductSumSTCovFun(Cs,Ct,k3=0,nugget_bounds=(0,2),k1_bounds=(0,2),k2_bounds=(0,2))
    SK = stk.SimpleKriging(hard_points,krig_points,'residual',spatial_column='geometry',temporal_column='time_val')    
    SK.build_distance_matrices(5000,np.inf)
    SK.estimate_variogram(Cst,spatial_lags,temporal_lags,options={'tol':0.0001})
    z_k,sigma_k = SK.krig_values(n_max=100,n_min=10)
    
    y_k = z_k + model.predict(X_k)
    sigma_k[np.isnan(sigma_k)] = np.std(SK.z_h)
    
    krig_points['log_val_transfer_kriged'] = y_k
    krig_points['sigma_log_val_transfer_kriged'] = sigma_k
    krig_points['val_transfer_kriged'] = np.exp(y_k + 0.5*sigma_k**2)
    krig_points['sigma_val_transfer_kriged'] = krig_points['val_transfer_kriged']**2*(np.exp(sigma_k**2)-1)
    krig_points['log_val_transfer_RF'] = model.predict(X_k) 
    cv_list.append(krig_points)
    
cv_df = pd.concat(cv_list).reset_index(drop=True)

# Calculate error metrics
cv_df['abs_error_log'] = np.abs((cv_df['log_val_transfer'] - cv_df['log_val_transfer_kriged']))
cv_df['abs_percent_error_log'] = cv_df['abs_error_log']/np.abs((cv_df['log_val_transfer']))
cv_df['abs_error_nonlog'] = np.abs((cv_df['val_transfer'] - cv_df['val_transfer_kriged']))
cv_df['abs_percent_error_nonlog'] = cv_df['abs_error_nonlog']/np.abs((cv_df['val_transfer']))

# Save cross-validation results
cv_outname = os.path.join(pv_folder,'property_value_cross_validation.gdb')
cv_df.to_file(cv_outname,layer=f'group_{group_idx}',driver='OpenFileGDB')

cv_object_outname = os.path.join(outfolder,f'group_{group_idx}_cross_validation.object')
with open(cv_object_outname,'wb') as f:
    pickle.dump(cv_df,f)
    f.close()
    
### *** RANDOM FOREST REGRESSION KRIGING PREDICTION *** ###

# Now train on all sales and predict for all single family homes in study area at each time point
X_h = sales[features].to_numpy()
y_h = sales[response_variable].to_numpy()

# Determine optimal hyperparameters for mean trend random forest model
param_grid = {'n_estimators':[200],'max_depth':[7,9,11],'max_features':[0.33,0.5,0.66]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=k, scoring='neg_mean_squared_error',n_jobs=n_cores)
grid_search.fit(X_h,y_h)
hyperparams = grid_search.best_params_

# Fit model to hard point data
model = RandomForestRegressor(**hyperparams).fit(X_h,y_h)

# Get residuals at hard points
z_h = y_h - model.predict(X_h)
sales['residual'] = z_h

# To reduce memory consumption, perform kriging in chunks of 300k
chunksize=300000
n_chunks = np.ceil(len(property_timepoints)/chunksize).astype(int)
krig_list = []

for i,krig_points in enumerate(np.array_split(property_timepoints,n_chunks)):
    
    print(f'\n*** Chunk {i+1} / {n_chunks} ***\n',flush=True)
    
    X_k = krig_points[features].to_numpy()

    # Estimate residuals at krig points via simple kriging (assume zero mean)
    Cs = stk.SphCovFun(a_bounds=(250,5000))
    Ct = stk.ExpCovFun(a=365)
    Cst = stk.ProductSumSTCovFun(Cs,Ct,k3=0,nugget_bounds=(0,2),k1_bounds=(0,2),k2_bounds=(0,2))
    SK = stk.SimpleKriging(sales,krig_points,'residual',spatial_column='geometry',temporal_column='time_val')    
    SK.build_distance_matrices(5000,np.inf)
    SK.estimate_variogram(Cst,spatial_lags,temporal_lags,options={'tol':0.0001})
    z_k,sigma_k = SK.krig_values(n_max=100,n_min=10)

    y_k = z_k + model.predict(X_k)
    sigma_k[np.isnan(sigma_k)] = np.std(SK.z_h)

    krig_points['log_val_transfer_kriged'] = y_k
    krig_points['sigma_log_val_transfer_kriged'] = sigma_k
    krig_points['val_transfer_kriged'] = np.exp(y_k + 0.5*sigma_k**2)
    krig_points['sigma_val_transfer_kriged'] = krig_points['val_transfer_kriged']**2*(np.exp(sigma_k**2)-1)
    krig_points['log_val_transfer_RF'] = model.predict(X_k)
    
    krig_list.append(krig_points)
    gc.collect()
    
kriged_df = pd.concat(krig_list)
kriged_df = kriged_df.drop(columns='geometry')
kriged_df = kriged_df.sort_values(by=['building_id','date']).reset_index(drop=True)

### *** SAVE RESULTS *** ###
kriged_outname = os.path.join(outfolder,f'group_{group_idx}_property_values_kriged.csv')
kriged_df.to_csv(kriged_outname,index=False)

covfun_outname = os.path.join(outfolder,f'group_{group_idx}_covariance_function.object')
with open(covfun_outname,'wb') as f:
    pickle.dump(SK.Cst,f)
    f.close()
    
variogram_outname = os.path.join(outfolder,f'group_{group_idx}_empirical_variogram.object')
with open(variogram_outname,'wb') as f:
    pickle.dump(SK.empirical_variogram,f)
    f.close()