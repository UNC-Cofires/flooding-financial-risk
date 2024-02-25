import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
import matplotlib.pyplot as plt
import scipy.stats as stats
import dependence_modeling as dm
import os
import pickle

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

def unround_number(x,multiple=1000):
    """
    Many variables are round to the nearest multiple of 10, 100, 1000, etc. This can cause values to be 
    artificially clustered around certain values. Because we want to fit continuous distributions, we will 
    add some noise to the rounded values to "unround" them. 
    
    param: x: rounded monetary value
    param: multiple: multiple of 10 that values are rounded to (i.e., 10, 100, 1000, etc.)
    returns: "unrounded" value of x
    """
    n = len(x)
    sign = np.sign(x)
    x = np.abs(x)

    r = stats.uniform(loc=-0.5).rvs(n)
    x += multiple*r
    x = sign*x
    
    return(x)

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

### *** SETUP FOLDERS *** ###

# Specify current working directory
pwd = os.getcwd()

# Specify output directory for estimated distributions
outfolder = os.path.join(pwd,dt.datetime.today().strftime('%Y-%m-%d_distributions'))
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)
    
# *** PROCESS HMDA MORTGAGE ORIGINATION DATA *** #

# Read in HMDA data
hmda_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/HMDA/hmda_nc_originations_1990-2022_clean.csv'
hmda_df = pd.read_csv(hmda_path,dtype={'census_tract':str,'county_code':str},index_col=0)

# Get mortgages written for purchase of primary residence, single-family homes
hmda_df = hmda_df[hmda_df['loan_purpose']=='Home purchase']
hmda_df = hmda_df[hmda_df['occupancy_type']=='Primary residence']
hmda_df = hmda_df[hmda_df['dwelling_type']=='Single Family (1-4 Units)']

# Drop entries that are missing census tract
hmda_df = hmda_df[~hmda_df['census_tract'].isna()]

# Drop pre-1992 mortgages since these can't easily be matched back to the census tract level
hmda_df = hmda_df[hmda_df['year'] >= 1992]

# Read in census tract boundaries for available census years

census_dir = "/proj/characklab/flooddata/NC/multiple_events/geospatial_data/TIGER"
tract_dict = {}

for year in [1990,2000,2010,2020]:
    tracts = gpd.read_file(os.path.join(census_dir,f'nc_{year}_census_tracts_clean'))
    census_tract_ids = tracts['GEOID'].to_list()
    tract_dict[year] = census_tract_ids
    
# Define the census tract year used for each annual HMDA data release
hmda_df['census_year'] = hmda_df['year'].apply(hmda_census_year)

# Drop entries that don't match to a census tract
n_begin = len(hmda_df)
for census_year in hmda_df['census_year'].unique():
    m1 = (hmda_df['census_year']==census_year)
    m2 = (~hmda_df['census_tract'].isin(tract_dict[census_year]))
    hmda_df = hmda_df[~(m1&m2)]
    
n_end = len(hmda_df)
percent_match = np.round(100*(n_end/n_begin),2)

print(f'Matched {n_end} / {n_begin} ({percent_match}%) loans to census tracts.',flush=True)

# "Unround" income and loan amount columns (which are rounded to nearest 1000)
hmda_df['loan_amount'] = unround_number(hmda_df['loan_amount'],multiple=1000)
hmda_df['income'] = unround_number(hmda_df['income'],multiple=1000)

# *** ADJUST MONETARY AMOUNTS FOR INFLATION *** #

# Read in data on inflation over time (as measured by consumer price index)
inflation_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/inflation_measures.csv'
inflation = pd.read_csv(inflation_path)
inflation['DATE'] = pd.to_datetime(inflation['DATE'])
inflation['year'] = inflation['DATE'].dt.year
inflation = inflation[['year','USACPIALLMINMEI']].groupby('year').mean().reset_index()

# Read in data on mortgage rates over time (as measured by the average 30-year fixed rate)
average_rate_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/interest_rates/MORTGAGE30US.csv'
average_rate = pd.read_csv(average_rate_path)
average_rate['DATE'] = pd.to_datetime(average_rate['DATE'])
average_rate['MORTGAGE30US'] = pd.to_numeric(average_rate['MORTGAGE30US'],errors='coerce')
average_rate['year'] = average_rate['DATE'].dt.year
annual_average_rate = average_rate[['year','MORTGAGE30US']].groupby('year').mean().reset_index()

# Joint mortgage rate info to dataframe
hmda_df = pd.merge(hmda_df,annual_average_rate,how='left',on='year')

# Convert borrower income and loan amount to 2020 USD
reference_year = 2020
reference_cpi = inflation[inflation['year']==reference_year]['USACPIALLMINMEI'].values[0]
inflation['multiplier'] = reference_cpi/inflation['USACPIALLMINMEI']

hmda_df = pd.merge(hmda_df,inflation,how='left',on='year')
hmda_df['nominal_loan_amount'] = hmda_df['loan_amount'].copy()
hmda_df['nominal_income'] = hmda_df['income'].copy()
hmda_df['loan_amount'] = hmda_df['loan_amount']*hmda_df['multiplier']
hmda_df['income'] = hmda_df['income']*hmda_df['multiplier']

# Drop most extreme values of income and/or loan amount
# (some of these were likely coded wrong)
hmda_df = drop_extreme_values(hmda_df,'income',alpha=0.001)
hmda_df = drop_extreme_values(hmda_df,'loan_amount',alpha=0.001)

# *** ESTIMATE DEPENDENCE OF LTV & RATE SPREAD ON INCOME & LOAN AMOUNT *** ###

# Interest rate and LTV at origination is only available from 2018 onwards
# Use this period to estimate the joint distribution of LTV and rate spread conditional on income and loan amount
hmda_recent = hmda_df[hmda_df['year']>=2018]

# Rates increased rapidly over the course of 2022
# This means that the rate on your mortgage dependend strongly on 
# the month of the year in which the mortgage was acquired
#
# Because we don't have information on the exact date of HMDA loans, and are comparing to the 
# average rate over the entire year, this has the effect of increasing the variance of the rate spread distribution.
#
# For this reason, I'm excluding mortgages originated in 2022 from rate spread calculation. 
hmda_recent = hmda_recent[hmda_recent['year'] != 2022]
hmda_df = hmda_df[hmda_df['year'] != 2022]

# Only consider 30-year fixed rate mortgages (~90% fall into this category)
hmda_recent = hmda_recent[hmda_recent['loan_term']==360]

# Make LTV numeric, and convert from percentage to ratio
# Note that 5-10% of mortgages have LTV > 1.0 at origination.
# Perhaps this is for financing repairs on a newly-bought home that's a fixer-upper?
hmda_recent['loan_to_value_ratio'] = hmda_recent['loan_to_value_ratio'].astype(float)/100

# Drop most extreme values of interest rate and LTV
hmda_recent = drop_extreme_values(hmda_recent,'interest_rate',alpha=0.001)
hmda_recent = drop_extreme_values(hmda_recent,'loan_to_value_ratio',alpha=0.001)

# Calculate borrower rate spread versus the yearly average for 30-year fixed rate mortgages
# (Note that the HMDA dataset already includes a "rate_spread" column, but it's not clear what comparator they use)
hmda_recent['rate_spread_30YF'] = hmda_recent['interest_rate'] - hmda_recent['MORTGAGE30US']

# Model joint distribution of income, loan amount, LTV, and rate spread using empirical marginals + guassian copula
# Assume that the dependence between these variables doesn't vary within the state
# (i.e., LTV and rate spread are just as correlated in Raleigh as they are in New Bern)

variables = ['income','loan_amount','loan_to_value_ratio','rate_spread_30YF']
marginals = {}

for var in variables:
    marginals[var] = dm.empirical_distribution(hmda_recent[var].to_numpy())

LTV_rate_depmod = dm.DependenceModel(marginals)
LTV_rate_depmod.fit_dependence(hmda_recent[variables])

# Save results
outname = os.path.join(outfolder,'LTV_rate_depmod.object')
with open(outname,'wb') as f:
    pickle.dump(LTV_rate_depmod,f)
    f.close()
    
# *** ESTIMATE MARGINAL DISTRIBUTIONS OF INCOME & LOAN AMOUNT STATEWIDE BY YEAR *** #

statelevel_income_loan_dist_by_year = {}

for year in hmda_df['year'].unique():
    
    marginals = {}

    # Get distribution of income / loan amount in real terms (i.e., adjusted for inflation to 2020 USD)
    income = hmda_df[hmda_df['year']==year]['income'].to_numpy()
    loan_amount = hmda_df[hmda_df['year']==year]['loan_amount'].to_numpy()
    marginals['income'] = dm.empirical_distribution(income)
    marginals['loan_amount'] = dm.empirical_distribution(loan_amount)

    # Also get distribution of income / loan amount in nominal terms
    nominal_income = hmda_df[hmda_df['year']==year]['nominal_income'].to_numpy()
    nominal_loan_amount = hmda_df[hmda_df['year']==year]['nominal_loan_amount'].to_numpy()
    marginals['nominal_income'] = dm.empirical_distribution(nominal_income)
    marginals['nominal_loan_amount'] = dm.empirical_distribution(nominal_loan_amount)
    
    statelevel_income_loan_dist_by_year[year] = marginals
    
# Save results
outname = os.path.join(outfolder,'statelevel_income_loan_dist_by_year.object')
with open(outname,'wb') as f:
    pickle.dump(statelevel_income_loan_dist_by_year,f)
    f.close()
    
# *** ESTIMATE MARGINAL DISTRIBUTIONS OF INCOME & LOAN AMOUNT BY YEAR & TRACT *** #

tractlevel_income_loan_dist_by_year = {}

for year in hmda_df['year'].unique():
    
    hmda_year = hmda_df[hmda_df['year']==year]
    tract_dict = {}
    
    for tract in np.sort(hmda_year['census_tract'].unique()):
    
        marginals = {}
        
        m = (hmda_year['census_tract']==tract)
        n_obs = np.sum(m)
        marginals['n_obs'] = n_obs
        
        if n_obs > 1:
            
            # Get distribution of income / loan amount in real terms (i.e., adjusted for inflation to 2020 USD)
            income = hmda_year[m]['income'].to_numpy()
            loan_amount = hmda_year[m]['loan_amount'].to_numpy()
            marginals['income'] = dm.empirical_distribution(income)
            marginals['loan_amount'] = dm.empirical_distribution(loan_amount)

            # Also get distribution of income / loan amount in nominal terms
            nominal_income = hmda_year[m]['nominal_income'].to_numpy()
            nominal_loan_amount = hmda_year[m]['nominal_loan_amount'].to_numpy()
            marginals['nominal_income'] = dm.empirical_distribution(nominal_income)
            marginals['nominal_loan_amount'] = dm.empirical_distribution(nominal_loan_amount)
            
        else:
                        
            marginals['income'] = hmda_year[m]['income'].values[0]
            marginals['loan_amount'] = hmda_year[m]['loan_amount'].values[0]
            marginals['nominal_income'] = hmda_year[m]['nominal_income'].values[0]
            marginals['nominal_loan_amount'] = hmda_year[m]['nominal_loan_amount'].values[0]
            
        tract_dict[tract] = marginals
            
    tractlevel_income_loan_dist_by_year[year] = tract_dict
    
#Save results
outname = os.path.join(outfolder,'tractlevel_income_loan_dist_by_year.object')
with open(outname,'wb') as f:
    pickle.dump(tractlevel_income_loan_dist_by_year,f)
    f.close()
    
### *** SAVE DATA USED TO CREATE DISTRIBUTIONS *** ###

# Drop specific columns from HMDA data before saving
# (these ones can't be trusted and are mostly missing for older time periods, so best to leave them out)
cols_to_drop = ['rate_spread','property_value','loan_to_value_ratio','debt_to_income_ratio','interest_rate','loan_term']
hmda_df = hmda_df.drop(columns=cols_to_drop)
outname = os.path.join(outfolder,'hmda_nc_home_purchase_mortgage_originations.csv')
hmda_df.to_csv(outname,index=False)

# Also save more recent data used to characterize LTV and rate spread distribution
outname = os.path.join(outfolder,'hmda_nc_home_purchase_mortgage_originations_recent.csv')
hmda_recent.to_csv(outname,index=False)