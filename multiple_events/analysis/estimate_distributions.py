import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
import matplotlib.pyplot as plt
import scipy.stats as stats
import pyarrow as pa
import pyarrow.parquet as pq
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

def monthly_payment(r,N,P):
    """
    This function calculates the monthly payment on a fixed-rate, fully amortizing loan. 
    See: https://en.wikipedia.org/wiki/Mortgage_calculator
    
    param: r: annual interest rate on loan (expressed as a percentage from 0-100)
    param: N: loan term (number of monthly payments)
    param: P: loan principal (initial unpaid balance)
    return: c: minimum monthly payment on loan
    """
    # Convert the annual interest rate to a monthly interest rate, and make decimal rather than percent
    r = (r/100)/12
    
    if r == 0:
        c = P/N
    else:
        c = r*P/(1-(1+r)**(-N))
    return(c)

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

# Get mortgages written for purchase or refinancing of primary residence, single-family homes
hmda_df = hmda_df[hmda_df['loan_purpose'].isin(['Home purchase','Refinancing'])]
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

# *** PROCESS FANNIE MAE & FREDDIE MAC MORTGAGE ORIGINATION DATA *** ###
gse_path = '/proj/characklab/flooddata/NC/multiple_events/financial_data/GSEs/fannie_freddie_SF_loan_level_originations.parquet'
table = pq.read_table(gse_path,use_pandas_metadata=True)
gse_df = table.to_pandas()

# Get mortgages written for purchase or refinancing of primary residence, single-family homes
gse_df = gse_df[gse_df['occupancy_status']=='Primary']

# "Unround" loan amount column (which was rounded to the nearest 1000)
gse_df['loan_amount'] = unround_number(gse_df['loan_amount'],multiple=1000)

# Make LTV and DTI on a 0-1 scale instead of percentages
gse_df['oDTI'] = gse_df['oDTI']/100
gse_df['oLTV'] = gse_df['oLTV']/100

# *** ESTIMATE NON-MORTGAGE DEBT OBLIGATIONS *** #

# Interest rate and DTI at origination is only available from 2018 onwards
# Use this period to estimate ratio of mortgage to non-mortgage monthly debt payments 
hmda_recent = hmda_df[hmda_df['year']>=2018]

# Calculate monthly mortgage payment
vect_monthly_payment = np.vectorize(monthly_payment)
hmda_recent['mortgage_payment'] = vect_monthly_payment(hmda_recent['interest_rate'],hmda_recent['loan_term'],hmda_recent['loan_amount'])

# Calculate monthly non-mortgage debt obligations from DTI, income, and mortgage payment
hmda_recent['DTI'] = pd.to_numeric(hmda_recent['debt_to_income_ratio'],errors='coerce')/100
hmda_recent['DTI'] = unround_number(hmda_recent['DTI'],multiple=0.01)
hmda_recent['monthly_income'] = hmda_recent['income']/12
hmda_recent['monthly_debt_obligations'] = np.maximum(hmda_recent['monthly_income']*hmda_recent['DTI'],hmda_recent['mortgage_payment'])

# Drop most extreme values
hmda_recent = drop_extreme_values(hmda_recent,'monthly_debt_obligations',alpha=0.001)
hmda_recent = drop_extreme_values(hmda_recent,'loan_amount',alpha=0.001)
hmda_recent = drop_extreme_values(hmda_recent,'income',alpha=0.001)

# Calculate proportion of total monthly debt payments that go towards mortgage
hmda_recent['mortgage_debt_fraction'] = hmda_recent['mortgage_payment']/hmda_recent['monthly_debt_obligations']

# Get distribution of mortgage debt fraction for middle 80% of borrowers 
# (exclude extremes where people have very small mortgage loans or huge non-mortgage debts)
x = hmda_recent['mortgage_debt_fraction']
alpha=0.1
x = x[(x>=x.quantile(alpha/2))&(x<=x.quantile(1-alpha/2))]
LB = int(np.round(x.min()*100))
UB = int(np.round(x.max()*100))

print(f'For {int(100*(1-alpha))}% of borrowers, mortgage payments make up {LB}-{UB}% of their total monthly debt obligations.',flush=True)

# Using distribution estimated from HMDA data, draw for mortgage debt fraction in Fannie / Freddie data
mortgage_fraction_dist = dm.empirical_distribution(x.to_numpy())
gse_df['mortgage_debt_fraction'] = mortgage_fraction_dist.rvs(len(gse_df))

# Calculate income of Fannie / Freddie borrowers using DTI, mortgage payment, and simulated mortgage debt fraction
gse_df['mortgage_payment'] = vect_monthly_payment(gse_df['interest_rate'],gse_df['loan_term'],gse_df['loan_amount'])
gse_df['monthly_debt_obligations'] = gse_df['mortgage_payment']/gse_df['mortgage_debt_fraction']
gse_df['monthly_income'] = gse_df['monthly_debt_obligations']/gse_df['oDTI']
gse_df['income'] = 12*gse_df['monthly_income']

# *** ESTIMATE DEPENDENCE OF LTV, DTI & RATE SPREAD ON INCOME & LOAN AMOUNT *** ###

# Remove entries with LTV at origination of >100%
gse_df = gse_df[gse_df['oLTV']<=1.0]

# Stratify by loan term
# For home purchase, ~90% are 30-year loans
# For refinancing, ~55% are 30-year loans, ~20% are 15-year loans, and ~25% are other terms
purchase30 = gse_df[(gse_df['loan_purpose']=='Purchase')&(gse_df['loan_term']==360)]
refinance30 = gse_df[(gse_df['loan_purpose']=='Refinance')&(gse_df['loan_term']==360)]
refinance15 = gse_df[(gse_df['loan_purpose']=='Refinance')&(gse_df['loan_term']==180)]

# Select relevant benchmark to compare interest rate against
purchase30['rate_spread'] = purchase30['spread_vs_MORTGAGE30US']
refinance30['rate_spread'] = refinance30['spread_vs_MORTGAGE30US']
refinance15['rate_spread'] = refinance15['spread_vs_MORTGAGE15US']

# Model joint distribution of income, loan amount, LTV, DTI, and rate spread using empirical marginals + guassian copula
# Assume that the dependence between these variables doesn't vary within the state
# (i.e., LTV and rate spread are just as correlated in Raleigh as they are in New Bern)
variables = ['income','loan_amount','oLTV','oDTI','rate_spread']

# Stratify joint distributions by year and loan type
statelevel_distributions_by_year = {}

for year in np.arange(1999,2020):

    p30 = purchase30[purchase30['origination_year']==year]
    r30 = refinance30[refinance30['origination_year']==year]
    r15 = refinance30[refinance30['origination_year']==year]

    p30_marginals = {}
    r30_marginals = {}
    r15_marginals = {}

    for var in variables:
        p30_marginals[var] = dm.empirical_distribution(p30[var].to_numpy())
        r30_marginals[var] = dm.empirical_distribution(r30[var].to_numpy())
        r15_marginals[var] = dm.empirical_distribution(r15[var].to_numpy())

    print(f'\n#---------- {year}: Purchase30 ----------#\n',flush=True)

    p30_depmod = dm.DependenceModel(p30_marginals)
    p30_depmod.fit_dependence(p30[variables])

    print(f'\n#---------- {year}: Refinance30 ----------#\n',flush=True)

    r30_depmod = dm.DependenceModel(r30_marginals)
    r30_depmod.fit_dependence(r30[variables])

    print(f'\n#---------- {year}: Refinance15 ----------#\n',flush=True)

    r15_depmod = dm.DependenceModel(r15_marginals)
    r15_depmod.fit_dependence(r15[variables])
    
    statelevel_distributions_by_year[year] = {'purchase30':p30_depmod,'refinance30':r30_depmod,'refinance15':r15_depmod}
    
### *** SAVE RESULTS AND DATA USED TO PARAMETERIZE DISTRIBUTIONS *** ###
outname = os.path.join(outfolder,'statelevel_distributions_by_year.object')
with open(outname,'wb') as f:
    pickle.dump(statelevel_distributions_by_year,f)
    f.close()
    
outname = os.path.join(outfolder,'hmda_mortgage_originations.csv')
hmda_df.to_csv(outname)

outname = os.path.join(outfolder,'gse_mortgage_originations.parquet')
gse_df.to_parquet(outname)