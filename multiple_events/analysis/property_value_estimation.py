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

# Clip to study area (include sales within the buffer zone as hard points to improve kriging at edges)
sales = sales[sales.intersects(buffered_study_area)]
sales['study_area'] = sales.intersects(study_area).astype(int)
homes = homes[homes.intersects(study_area)]

### *** SPECIFY TIMEPOINTS OF PROPERTY VALUE ESTIMATION *** ###

# Get dates corresponding to midpoint of each quarter between 1995 and 2020
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
property_timepoints['month'] = property_timepoints['date'].dt.month
property_timepoints['time_val'] = (property_timepoints['date'] - pd.Timestamp('1990-01-01')).dt.days

# Specify values of time dependent predictor variables (i.e., home price index and seasonal trend)
home_price_index_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/property_value_inputs/nc_daily_interpolated_home_price_index.csv'
seasonal_component_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/property_value_inputs/nc_home_price_seasonal_component.csv'
home_price_index = pd.read_csv(home_price_index_path)
home_price_index['date'] = pd.to_datetime(home_price_index['date'])
seasonal_component = pd.read_csv(seasonal_component_path)

property_timepoints = pd.merge(property_timepoints,home_price_index[['date','home_price_index']],on='date',how='left')
property_timepoints = pd.merge(property_timepoints,seasonal_component[['month','seasonal_component']],on='month',how='left')

# Log-transform specified variables
sales[f'log_val_transfer'] = np.log(sales['val_transfer'])
log_transform_variables = ['HTD_SQ_FT','parcel_sq_ft','median_hh_income']

for var in log_transform_variables:
    sales[f'log_{var}'] = np.log(sales[var])
    property_timepoints[f'log_{var}'] = np.log(property_timepoints[var])
    

    
### *** RANDOM FOREST REGRESSION KRIGING CROSS VALIDATION *** ###

response_variable = 'log_val_transfer'
features = ['log_HTD_SQ_FT','log_parcel_sq_ft','YEAR_BUILT','log_median_hh_income','home_price_index','seasonal_component']

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
    z_k,sigma_k = SK.krig_values(n_max=250,n_min=50)
    
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

# To reduce memory consumption, perform kriging in chunks
n_chunks = 30
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
    z_k,sigma_k = SK.krig_values(n_max=250,n_min=50)

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