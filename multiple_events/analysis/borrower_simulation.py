import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.stats as stats
import scipy.interpolate as interp
import pyarrow as pa
import pyarrow.parquet as pq
import dependence_modeling as dm
import mortgage_model as mm
from copy import deepcopy
import pickle
import os
import sys
import time

### *** HELPER FUNCTIONS *** ###

def drop_extreme_values(df,column,alpha=0.001):
    """
    Thus function will drop rows of a data frame that includes extremely high or low values
    as defined by the 100*(alpha/2)% and 100*(1-alpha/2)% percentiles.
    
    param: df: pandas dataframe
    param: column: column of interest
    param: alpha: significance level used to define a value as extremely high or low (should be small)
    """
    min_value = df[column].quantile(alpha/2)
    max_value = df[column].quantile(1-alpha/2)

    m1 = (df[column] < min_value)
    m2 = (df[column] > max_value)
    m = ~(m1|m2)
    
    return(df[m])

def format_elapsed_time(seconds):
    seconds = int(np.round(seconds))
    hours = seconds // 3600
    seconds = seconds - hours*3600
    minutes = seconds // 60
    seconds = seconds - minutes*60
    return(f'{hours}h:{minutes:02d}m:{seconds:02d}s')

### *** SET UP FOLDERS AND ENVIRONMENT *** ###

# Get index of county to simulate
county_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])

# Read in command-line arguments
replication_number = int(sys.argv[1])
damage_cost_multiplier = float(sys.argv[2])
property_value_multiplier = float(sys.argv[3])
repair_rate_multiplier = float(sys.argv[4])

if damage_cost_multiplier == property_value_multiplier == repair_rate_multiplier == 1:
    suffix = 'base_case'
else:
    suffix = f'DC_{damage_cost_multiplier:.2f}_PV_{property_value_multiplier:.2f}_RR_{repair_rate_multiplier:.2f}'

# Specify current working directory
pwd = os.getcwd()

# Specify output directory for model runs
outfolder = os.path.join(pwd,f'mortgage_borrower_simulation_{suffix}/replicate_{replication_number}')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)
    
### *** DEFINE START AND END POINTS OF SIMULATION *** ###
start_period = '1990-01'
end_period = '2019-12'

start_period = pd.to_datetime(start_period).to_period('M')
end_period = pd.to_datetime(end_period).to_period('M')

periods = pd.period_range(start_period,end_period)

period_midpoint = periods.start_time + 0.5*(periods.end_time - periods.start_time)
period_midpoint = pd.to_datetime(period_midpoint.date)
    
### *** GEOSPATIAL DATA SOURCES *** ###

crs = 'EPSG:32617'

# Read in data on NC counties
counties_path = '/proj/characklab/flooddata/NC/multiple_events/geospatial_data/NC_counties'
counties = gpd.read_file(counties_path).to_crs(crs)
counties = counties[['FIPS','County','geometry']].rename(columns={'FIPS':'countyCode','County':'countyName'})
counties = counties.sort_values(by='countyCode').reset_index(drop=True)

# Get name / FIPS code of county of interest
county_code = counties.loc[county_idx,'countyCode']
county_name = counties.loc[county_idx,'countyName']

# Create folder for results from county
county_folder = os.path.join(outfolder,county_name)
if not os.path.exists(county_folder):
    os.makedirs(county_folder,exist_ok=True)

print(f'#---------- {county_name} ----------#\n',flush=True)

### *** DAMAGE EXPOSURE DATA *** ###

# Read in data on damage exposure of single family homes in county
damage_dir = os.path.join(pwd,'2024-08-25_damage_estimates_by_county')

properties_path = os.path.join(damage_dir,county_name,f'{county_name}_homes.parquet')
properties = pq.read_table(properties_path,use_pandas_metadata=True).to_pandas()

exposure_path = os.path.join(damage_dir,county_name,f'{county_name}_exposure_timeseries.parquet')
damage_exposure = pq.read_table(exposure_path,use_pandas_metadata=True).to_pandas()
damage_exposure = damage_exposure[damage_exposure['period'].isin(periods)]

### *** PROPERTY VALUE DATA *** ###

# Read in information on property value estimates
pv_dir = os.path.join(pwd,'property_value_estimates_by_county')
pv_path = os.path.join(pv_dir,f'{county_name}/{county_name}_property_values_kriged.parquet')
property_values = pq.read_table(pv_path,use_pandas_metadata=True).to_pandas()

# There's a tiny number of properties we don't have property value estimates for, which we'll drop from the analysis
m = properties['building_id'].isin(property_values['building_id'])
n_prop = len(m)
n_drop = np.sum(~m)
properties = properties[m]
print(f'Dropped {n_drop} / {n_prop} ({np.round(100*n_drop/n_prop,3)}%) properties.',flush=True)

# Save to file
outname = os.path.join(county_folder,f'{county_name}_properties.parquet')
properties.to_parquet(outname)

# Property values were estimated on a quarterly basis
# Interpolate the estimated values to be on a monthly basis
# (This step can take awhile if the number of buildings is large)

print('Interpolating monthly property values from quarterly estimates.\n',flush=True)

property_values = property_values.rename(columns={'period':'quarter'})
interp_tvals = np.array((period_midpoint - period_midpoint.min()).days)

n_properties = len(properties)
n_periods = len(periods)

building_id_array = np.zeros(n_properties*n_periods)
date_array = np.empty(n_properties*n_periods,dtype='object')
pv_array = np.zeros(n_properties*n_periods)

period_midpoint_str = period_midpoint.strftime('%Y-%m-%d')

for i,building_id in enumerate(properties['building_id'].unique()):
    
    start_idx = i*n_periods
    end_idx = (i+1)*n_periods
    
    building_id_array[start_idx:end_idx] = building_id
    date_array[start_idx:end_idx] = period_midpoint_str
    
    temp = property_values[property_values['building_id']==building_id]
    pv_date = temp['date']
    pv_obs = temp['val_transfer_kriged']
    pv_tvals = np.array((pv_date - period_midpoint.min()).dt.days)
    
    interp_func = interp.interp1d(pv_tvals,pv_obs,kind='linear',fill_value='extrapolate')
    
    pv_array[start_idx:end_idx] = interp_func(interp_tvals)
    
pv_timeseries = pd.DataFrame({'building_id':building_id_array,'period':date_array,'property_value':pv_array})
pv_timeseries['period'] = pd.to_datetime(pv_timeseries['period']).dt.to_period('M')

### *** MORTGAGE ORIGINATION DATA *** ###

# Read in data on mortgage originations
originations_dir = os.path.join(pwd,'2024-07-19_distributions')
originations_path = os.path.join(originations_dir,'hmda_mortgage_originations.csv')
originations = pd.read_csv(originations_path,index_col=0,dtype={'county_code':str,'census_tract':str})
originations = originations.rename(columns={'census_tract':'censusTract','county_code':'countyCode','census_year':'censusYear'})
originations = originations[originations['countyCode']==county_code].reset_index(drop=True)

# Randomly assign month of origination since we only have info on year
originations['month'] = np.random.randint(1,12+1,size=len(originations))
originations['period'] = originations['year'].astype(str) + '-' + originations['month'].apply(lambda x: '{:02d}'.format(x))
originations['period'] = pd.to_datetime(originations['period']).dt.to_period(freq='M')
originations['year'] = originations['period'].dt.year
originations = originations[['period','year','state','countyCode','censusTract','censusYear','loan_purpose','loan_amount','income']]
originations = originations[originations['period'].isin(periods)]
originations = originations.sort_values(by='period').reset_index(drop=True)

# Assign loan term
# Assume 100% of home purchase loans are 30-year mortgages
# Assume that 2/3 refinance loans are 30-year, while 1/3 are 15-year
originations['loan_purpose'] = originations['loan_purpose'].apply(lambda x: 'refinance' if x == 'Refinancing' else 'purchase')
originations['loan_term'] = originations['loan_purpose'].apply(lambda x: np.random.choice([360,180],p=[2/3,1/3]) if x == 'refinance' else 360)

# Read in data on joint distribution of LTV, DTI, etc. at origination by year
with open(os.path.join(originations_dir,'statelevel_distributions_by_year.object'), 'rb') as f:
    jointdist_by_year = pickle.load(f)

### *** MORTGAGE REPAYMENT DATA *** ###

# Read in data on time to mortgage prepayment
survival_dir = os.path.join(pwd,'2024-03-31_loan_survival_analysis')
p30_path = os.path.join(survival_dir,'purchase30_survival_params.csv')
r30_path = os.path.join(survival_dir,'refinance30_survival_params.csv')
r15_path = os.path.join(survival_dir,'refinance15_survival_params.csv')
p30_surv = pd.read_csv(p30_path)
r30_surv = pd.read_csv(r30_path)
r15_surv = pd.read_csv(r15_path)

# From outputs of loan survival analysis, create functions describing monthly prepayment prob
# as a function of (1) loan age and (2) spread versus current market rate, stratified by structure of loan. 

prepayment_profiles = {}

# Home purchase / 30-year

t = p30_surv['time'].to_numpy()
S = p30_surv['surv'].to_numpy()
beta = p30_surv['rate_spread_coeff'].values[0]
t_cutoff = 240

hazard_rate,monthly_prob = mm.prepayment_hazard(t,S,beta=beta,t_cutoff=t_cutoff)

prepayment_profiles['purchase30'] = monthly_prob

# Refinance / 30-year

t = r30_surv['time'].to_numpy()
S = r30_surv['surv'].to_numpy()
beta = r30_surv['rate_spread_coeff'].values[0]
t_cutoff = 240

hazard_rate,monthly_prob = mm.prepayment_hazard(t,S,beta=beta,t_cutoff=t_cutoff)

prepayment_profiles['refinance30'] = monthly_prob

# Refinance / 15-year

t = r15_surv['time'].to_numpy()
S = r15_surv['surv'].to_numpy()
beta = r15_surv['rate_spread_coeff'].values[0]
t_cutoff = 180

hazard_rate,monthly_prob = mm.prepayment_hazard(t,S,beta=beta,t_cutoff=t_cutoff)

prepayment_profiles['refinance15'] = monthly_prob

### *** INTEREST RATE DATA *** ###

# Read in data on mortgage rates over time

rate30_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/interest_rates/MORTGAGE30US.csv'
rate15_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/interest_rates/MORTGAGE15US.csv'

rate30 = pd.read_csv(rate30_path)
rate30['DATE'] = pd.to_datetime(rate30['DATE'])
rate30['MORTGAGE30US'] = pd.to_numeric(rate30['MORTGAGE30US'],errors='coerce')

rate30_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/interest_rates/MORTGAGE30US.csv'
rate15_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/interest_rates/MORTGAGE15US.csv'

rate30 = pd.read_csv(rate30_path)
rate30['DATE'] = pd.to_datetime(rate30['DATE'])
rate30['MORTGAGE30US'] = pd.to_numeric(rate30['MORTGAGE30US'],errors='coerce')

rate15 = pd.read_csv(rate15_path)
rate15['DATE'] = pd.to_datetime(rate15['DATE'])
rate15['MORTGAGE15US'] = pd.to_numeric(rate15['MORTGAGE15US'],errors='coerce')

rate30 = rate30.dropna()
rate15 = rate15.dropna()

rate30['period'] = rate30['DATE'].dt.to_period(freq='M')
rate30 = rate30[['period','MORTGAGE30US']].groupby('period').mean().reset_index()

rate15['period'] = rate15['DATE'].dt.to_period(freq='M')
rate15 = rate15[['period','MORTGAGE15US']].groupby('period').mean().reset_index()

market_rates = pd.merge(rate30,rate15,on='period',how='left')
market_rates = market_rates[market_rates['period'].isin(periods)].reset_index(drop=True)

# No data on average 15-year mortgage rates prior to August 1991
# For this period, impute the missing 15-year rate based on average spread between 30-year and 15-year rate
# This is usually ~50 basis points less than the 30-year rate
average_spread_15y_vs_30y = (market_rates['MORTGAGE15US'] - market_rates['MORTGAGE30US']).mean()
m = market_rates['MORTGAGE15US'].isna()
market_rates.loc[m,'MORTGAGE15US'] = market_rates.loc[m,'MORTGAGE30US'] + average_spread_15y_vs_30y
market_rates = market_rates.set_index('period')

# Specify interest rates on home repair loans / disaster recovery loans
# In the base case, assume equal to the average 30-year fixed mortgage rate
# In sensitivity analysis, assume equal to 50% of average 30-year fixed rate, since this approximates the SBA's below-market rate
repair_rate = repair_rate_multiplier*market_rates[f'MORTGAGE30US']

### *** INCOME GROWTH DATA *** ###

# Read in county-level data on annual personal income over time collected by the U.S. BEA

personal_income_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/personal_income/CAINC1_NC_1969_2022.csv'
personal_income = pd.read_csv(personal_income_path)
personal_income['GeoFIPS'] = personal_income['GeoFIPS'].str.strip(' "')
personal_income = personal_income[(personal_income['GeoFIPS'] == f'37{county_code}')&(personal_income['LineCode']==3)]
years = np.arange(1969,2023)
income = personal_income[years.astype(str)].iloc[0].values

# Income values are currently specified on an annual basis
# Interpolate them to be on a monthly basis assuming annual value corresponds to midpoint of year
prange = pd.period_range(1969,2022,freq='Y')

start_date = pd.Series([pd.Timestamp(p.start_time.date()) for p in prange])
end_date = pd.Series([pd.Timestamp(p.end_time.date()) for p in prange])
mid_date = start_date + 0.5*(end_date - start_date)
drange = pd.date_range(mid_date.min(),mid_date.max())

t_interp = np.array((drange - mid_date.min()).days)
t_hard = np.array((mid_date - mid_date.min()).dt.days)

interp_func = interp.interp1d(t_hard,income,kind='linear')
interpolated_income = interp_func(t_interp)

interp_periods = drange.to_period(freq='M')

income_df = pd.DataFrame({'period':interp_periods,'per_capita_income':interpolated_income})
income_df = income_df.groupby('period').mean().reset_index()
income_df = income_df[income_df['period'].isin(periods)].reset_index(drop=True)

income_growth = income_df.set_index('period')['per_capita_income']

# Convert to a growth rate (i.e., monthly % change in income)
income_growth = (income_growth.shift(-1)/income_growth).ffill() - 1

# *** SPECIFY JOINT DISTRIBUTIONS FOR PRE-1999 PERIOD *** ###

# GSE dataset has granular loan-level information by state going back to 1999
# HMDA dataset includes data on originations by tract going back to ~1992, with borrower income + loan amounts
# For years prior to 1999, we'll combine the following datasources to specify the joint distribution of income, LTV, DTI, etc. 
# (1) marginal distributions of income and loan amount from HMDA data
# (2) marginal distributions of LTV, DTI, and rate spread at origination from 1999 GSE data
# (3) gaussian copula fit to 1999 GSE data 

hmda_years = list(originations['period'].dt.year.unique())
missing_years = [x for x in hmda_years if x not in jointdist_by_year.keys()]

exclude_combos = []

for year in missing_years:
    
    year_dict = {}
    
    for loan_purpose,loan_term_years in zip(['purchase','refinance','refinance'],[30,30,15]):
        
        loan_term_months = 12*loan_term_years
        loan_type = f'{loan_purpose}{loan_term_years}'
        
        # Get originations corresponding to specific pre-1999 year and loan type
        m1 = (originations['period'].dt.year == year)
        m2 = (originations['loan_purpose'] == loan_purpose)
        m3 = (originations['loan_term'] == loan_term_months)
        hmda_df = originations[m1&m2&m3]
        
        # Don't try to fit distribution if there's less than 30 entries
        if len(hmda_df) >= 30:
        
            # Drop extreme values of income and loan amount that we don't want to have in modeled distribution
            hmda_df = drop_extreme_values(hmda_df,'income',alpha=0.01)
            hmda_df = drop_extreme_values(hmda_df,'loan_amount',alpha=0.01)

            # Copy 1999 joint distribution
            marginals = deepcopy(jointdist_by_year[1999][loan_type].marginals)
            R = deepcopy(jointdist_by_year[1999][loan_type].R)
            copula = deepcopy(jointdist_by_year[1999][loan_type].copula)

            # Update marginal distribution of income and loan amount, keeping other aspects of joint distribution the same
            marginals['income'] = dm.empirical_distribution(hmda_df['income'].to_numpy())
            marginals['loan_amount'] = dm.empirical_distribution(hmda_df['loan_amount'].to_numpy())

            # Create multivariate distribution from marginals + copula
            depmod = dm.DependenceModel(marginals)
            depmod.R = R
            depmod.copula = copula

            # Add to collection of joint distributions from given year
            year_dict[loan_type] = depmod
            
        else:
            # Don't try to model loans where there's very few entries 
            exclude_combos.append((year,loan_purpose,loan_term_months))
            
    # Add to collection of joint distributions by year
    jointdist_by_year[year] = year_dict
    
# Drop originations where the distribution of key variables is uncharacterized due to the tiny number of entries
mask = pd.Series([False]*len(originations))
for combo in exclude_combos:
    year,loan_purpose,loan_term = combo
    m = (originations['period'].dt.year == year)&(originations['loan_purpose'] == loan_purpose)&(originations['loan_term'] == loan_term)
    mask = mask|m
    
originations = originations[~mask].reset_index(drop=True)

n_drop = int(np.sum(mask))
p_drop = n_drop/len(mask)
print(f'Dropped {n_drop} / {len(mask)} ({np.round(p_drop*100,3)}%) loans due to poor characterization of key financial variables.\n',flush=True)

# Create a unique loan id for each origination consisting of county code, year, index of loan in that year
originations['loan_id'] = originations['countyCode'] + '-' + originations['year'].astype(str) + '-'
originations['loan_id'] += originations.groupby('loan_id').cumcount().astype(str)

### *** IF APPLICABLE, RESUME SIMULATION WHERE PREVIOUS RUN LEFT OFF *** ###

output_files = [x for x in os.listdir(county_folder) if 'simulation_output' in x]
output_files.sort()

if len(output_files) > 0:
    start_origination_year = 1 + int(output_files[-1].split('_')[-1].strip('.parquet'))
    
    # Read in record of building occupancy from last simulation snapshot
    # We'll use this to ensure we don't assign multiple concurrent loans to the same property
    property_availability_path = os.path.join(county_folder,f'{county_name}_property_availability.parquet')
    property_availability = pq.read_table(property_availability_path,use_pandas_metadata=True).to_pandas()
else:
    start_origination_year = originations['year'].min()
    property_availability = pv_timeseries.copy()
    
end_origination_year = originations['year'].max()
origination_years = np.arange(start_origination_year,end_origination_year+1)


### *** PERFORM DYNAMIC SIMULATION OF MORTGAGE BORROWER FINANCES WITHIN COUNTY *** ###

print('Simulating borrower financial conditions over time.',flush=True)

t0 = time.time()

for origination_year in origination_years:
    
    current_year_originations = originations[originations['year']==origination_year]
    summary_list = []
    failed_loans = []
    
    t1 = time.time()

    for origination in current_year_originations.to_dict(orient='records'):

        loan_id = origination['loan_id']
        origination_period = origination['period']
        income = origination['income']
        loan_purpose = origination['loan_purpose']
        loan_amount = origination['loan_amount']
        loan_term = origination['loan_term']

        geography_type = 'censusTract_' + str(origination['censusYear'])
        geographic_unit = origination['censusTract']

        # Select appropriate benchmark market rate
        loan_years = int(loan_term/12)
        market_rate = market_rates[f'MORTGAGE{loan_years}US']

        # Select appropriate monthly prepayment hazard function
        monthly_prepayment_prob = prepayment_profiles[f'{loan_purpose}{loan_years}']

        # Simulate DTI and rate spread at origination conditional on loan amount and income
        known_values = np.array([[income,loan_amount,np.nan,np.nan,np.nan,np.nan]])
        depmod = jointdist_by_year[origination_year][f'{loan_purpose}{loan_years}']

        # Because we simulate DTI and interest rate from a guassian copula, it is 
        # theoretically possible to end up with a DTI at origination that is less than the ratio
        # of the monthly mortgage payment / monthly income (which makes no sense). 
        # This happens extremely rarely, but if it does, try to redraw values up to 10 times. 
        # If it still doesn't work, discard the mortgage from the simulation. 

        keepgoing = True
        n_draws = 0

        while keepgoing:

            income,loan_amount,extra,oDTI,credit_score,rate_spread = depmod.conditional_simulation(known_values)[0]
            interest_rate = market_rate[origination_period] + rate_spread
            monthly_payment = mm.monthly_payment(interest_rate,loan_term,loan_amount)
            monthly_income = income/12

            n_draws += 1

            if monthly_payment/monthly_income <= oDTI:
                keepgoing = False
                success = True
            elif n_draws >= 10:
                keepgoing = False
                success = False

        # Assign mortgage to a specific property within census tract
        # We'll do this randomly, but use information on the joint distribution of LTV, rate spread, etc. 
        # to select a property that will give us a realistic LTV at origination
        potential_building_ids = properties[properties[geography_type]==geographic_unit]['building_id']
        potential_property_values = property_availability[property_availability['period']==origination_period]
        potential_property_values = potential_property_values[potential_property_values['building_id'].isin(potential_building_ids)]

        # Exclude properties that would give us LTV > 100% at origination
        potential_property_values['oLTV'] = loan_amount/potential_property_values['property_value']
        min_oLTV = depmod.marginals['oLTV'].ppf(0)
        max_oLTV = depmod.marginals['oLTV'].ppf(1)
        oLTV_mask = (potential_property_values['oLTV'] >= min_oLTV)&(potential_property_values['oLTV'] <= max_oLTV)
        potential_property_values = potential_property_values[oLTV_mask]

        if len(potential_property_values) == 0:
            success = False

        if success:

            # Calculate probability mass associated with potential properties (which differ in terms of LTV at origination) 
            potential_property_values['prob'] = depmod.marginals['oLTV'].pdf(potential_property_values['oLTV'].to_numpy())
            potential_property_values['prob'] = potential_property_values['prob']/potential_property_values['prob'].sum()
            potential_property_values['prob'] = np.minimum(np.maximum(potential_property_values['prob'].fillna(0),0),1)
            potential_property_values['prob'] = potential_property_values['prob']/potential_property_values['prob'].sum()

            # Randomly select property according to probability mass
            building_id = np.random.choice(potential_property_values['building_id'].to_numpy(),p=potential_property_values['prob'].to_numpy())
            
            # Get property value and damage exposure timeseries for selected property
            property_value = property_value_multiplier*pv_timeseries[pv_timeseries['building_id']==building_id].set_index('period')['property_value']
            property_damage_exposure = damage_cost_multiplier*damage_exposure[damage_exposure['building_id']==building_id].set_index('period').drop(columns='building_id')

            # Initialize borrower class and simulate repayment
            B = mm.MortgageBorrower(loan_id,building_id,origination_period,loan_purpose,loan_amount,loan_term,interest_rate,income,oDTI,credit_score)
            B.initialize_state_variables(property_value,market_rate,repair_rate,income_growth,property_damage_exposure,end_period=end_period)
            B.simulate_repayment(monthly_prepayment_prob)
            summary_list.append(B.summarize())

            # During the periods for which the was being repaid, mark the property as occupied so that we 
            # don't end up with multiple overlapping mortgages on the same property

            m = (property_availability['building_id']==B.building_id)&(property_availability['period'] >= B.periods[0])&(property_availability['period'] <= B.periods[-1])
            property_availability = property_availability[~m]

        else:
            failed_loans.append(loan_id)

    summary = pd.concat(summary_list).reset_index(drop=True)

    ### *** SAVE RESULTS *** ###
    
    # Save current state of building occupancy so that it can be resumed later
    outname = os.path.join(county_folder,f'{county_name}_property_availability.parquet')
    property_availability.to_parquet(outname)

    outname = os.path.join(county_folder,f'{county_name}_originations_{origination_year}.parquet')
    current_year_originations.to_parquet(outname)

    outname = os.path.join(county_folder,f'{county_name}_simulation_output_{origination_year}.parquet')
    summary.to_parquet(outname)

    outname = os.path.join(county_folder,f'{county_name}_failed_loans_{origination_year}.csv')
    failed_loan_df = pd.DataFrame({'loan_id':np.array(failed_loans)})
    failed_loan_df.to_csv(outname,index=False)
    
    ### *** PRINT UPDATE *** ###
    
    t2 = time.time()
    
    elapsed_time = format_elapsed_time(t2-t1)
    cumulative_elapsed_time = format_elapsed_time(t2-t0)
    
    n_current = len(current_year_originations)
    p_current = np.round(100*n_current/len(originations),1)

    progress = f'    Simulated {n_current} mortgages originated in {origination_year} ({p_current}% of total).'
    progress += f'Time elapsed: {elapsed_time} last iteration / {cumulative_elapsed_time} cumulative.'

    print(progress,flush=True)