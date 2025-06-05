import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.interpolate as interp
import dependence_modeling as dm
import pickle
import os

### *** HELPER FUNCTIONS *** ###

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
    
    # Calculate monthly payment
    c = r*P/(1-(1+r)**(-N))
     
    # Round up to nearest cent
    c = np.ceil(c*100)/100
    
    return(c)

def parameterize_lognormal_dist(mean,variance):
    """
    param: mean: desired mean of lognormal distribution
    param: variance: desired variance of lognormal distribution
    returns: d: lognormal distribution with desired mean and variance (instance of scipy.stats.lognorm)
    """

    mu = np.log(mean**2/np.sqrt(mean**2 + variance))
    sigma = np.sqrt(np.log(1+variance/mean**2))

    d = stats.lognorm(s=sigma,scale=np.exp(mu))
    
    return(d)

def geometric_brownian_motion(mu,sigma,S0,T):
    """
    param: mu: annualized return (continuously compounded)
    param: sigma: annualized volatility
    param: S0: value at time 0
    param: T: time elapsed in years
    returns: ST: distribution of value at time T (instance of scipy.stats.lognorm)
    """
    
    mean = S0*np.exp(mu*T)
    variance = S0**2*np.exp(2*mu*T)*(np.exp(sigma**2*T) - 1)
    ST = parameterize_lognormal_dist(mean,variance)
    return(ST)

def characterize_income_uncertainty(borrower,income_growth,sigma=0.05):
    """
    param: borrower: row of pandas dataframe containing characteristics for specific borrower
    param: income_growth: pandas series capturing monthly changes in average personal income for borrower's county 
    param: sigma: annualized volatility in income growth
    returns: ST: distribution of potential borrower income values (instance of scipy.stats.lognorm)
    """
    
    # Get origination date of borrower's mortgage
    # (This represents the last time that their income was directly observed)
    ti = borrower.period - borrower.loan_age
    
    # Get the current date
    tf = borrower.period
    
    # Get the time elapsed in years since origination
    T = ((tf - ti).n + 1) / 12
    
    # Get the average monthly change in income for the borrower's county during this period
    g = income_growth[(income_growth.index >= ti)&(income_growth.index < tf)]

    # Compute the expected "total return" or total percent growth in their income during this period
    R = np.prod(1+g) - 1
    
    # Base case value of annual income in current period
    S_bar = 12*borrower.monthly_income
    
    # Income at origination
    S0 = S_bar/(1+R)
    
    # Compute expected annualized return since origination (continuosly compounded)
    mu = (1/T)*np.log(S_bar/S0)
    
    # Calculate uncertainty in current estimate of income by assuming 
    # that income dynamics follow geometric brownian motion. This will
    # result in a log-normal distribution whose mean is equal to our base case
    # estimate of their income and whose variance is controlled by the assumed
    # volatility in income growth (sigma). 
    #
    # The amount of uncertainty will increase with the time since origination. 
    
    ST = geometric_brownian_motion(mu,sigma,S0,T)
    
    return(ST)

def characterize_damage_cost_uncertainty(borrower,damage_cost_conditional_variance):
    """
    param: borrower: row of pandas dataframe containing characteristics for specific borrower
    param: damage_cost_conditional_variance: dictionary containing conditional variance functions for each event
    returns: damage_dist: distribution of damage cost values (instance of scipy.stats.lognorm)
    """
    
    period_str = str(borrower.period)
    var_func = damage_cost_conditional_variance[period_str]

    mean = borrower.uninsured_damage
    variance = var_func(mean)

    damage_dist = parameterize_lognormal_dist(mean,variance)
    
    return(damage_dist)

def characterize_property_value_uncertainty(borrower):
    """
    param: borrower: row of pandas dataframe containing characteristics for specific borrower
    returns: pv_dist: distribution of potential property values (instance of scipy.stats.lognorm)
    """
    
    # Mean of log(P) from kriging
    mu = borrower.logpv_mu
    
    # Standard deviation of log(P) from kriging
    sigma = borrower.logpv_sigma
    
    pv_dist = stats.lognorm(s=sigma,scale=np.exp(mu))
    
    return(pv_dist)

def compute_sobol_indices(borrower,income_growth,damage_cost_conditional_variance,n_samples=2**16,sigma=0.05):
    """
    param: borrower: row of pandas dataframe containing characteristics for specific borrower
    param: income_growth: pandas series capturing monthly changes in average personal income for borrower's county 
    param: damage_cost_conditional_variance: dictionary containing conditional variance functions for damage costs in each event
    param: n_samples: number of monte carlo samples to draw when computing sobol indices. Must be a power of 2. 
    param: sigma: annualized volatility in income growth (user-defined assumption)
    returns: df: dataframe providing information on mean/sd of key inputs and outputs, as well as sobol indices
    returns: depmod: instance of DependenceModel class describing multivariate distribution of inputs 
    """
    
    marginals = {}
    
    # Define distributions of property value, income, and damage cost
    marginals['property_value'] = characterize_property_value_uncertainty(borrower)
    marginals['annual_income'] = characterize_income_uncertainty(borrower,income_growth,sigma=sigma)
    marginals['damage_cost'] = characterize_damage_cost_uncertainty(borrower,damage_cost_conditional_variance)
    
    # Create multivariate distribution
    depmod = dm.DependenceModel(marginals)
    
    # Define borrower-specific function describing whether they meet the criteria for 
    # strategic, cashflow, double-trigger, or any default for a given set of inputs
    # We'll later pass this function into scipy.stats.sobol_indices to do the heavy-lifting
    
    def default_func(x):
    
        property_value = x[0]
        annual_income = x[1]
        uninsured_damage = x[2]
        
        aLTV_cutoff = 1.0
        aDTI_cutoff = 0.45

        monthly_income = annual_income / 12

        repair_loan_payment = monthly_payment(borrower.repair_rate,360,uninsured_damage)

        aLTV = (borrower.unpaid_balance_on_all_loans + uninsured_damage) / property_value
        aDTI = (borrower.monthly_debt_obligations + repair_loan_payment) / monthly_income

        strategic = (aLTV > aLTV_cutoff)
        cashflow = (aDTI > aDTI_cutoff)
        double_trigger = strategic&cashflow
        any_default = (strategic|cashflow)

        outcomes = np.zeros((4,x.shape[1]))

        outcomes[0] = strategic.astype(float)
        outcomes[1] = cashflow.astype(float)
        outcomes[2] = double_trigger.astype(float)
        outcomes[3] = any_default.astype(float)

        return(outcomes)
    
    # Comput mean and standard deviation of key inputs and outputs
    input_samples = depmod.simulate_values(n_samples).T
    input_mean = input_samples.mean(axis=1)
    input_std = input_samples.std(axis=1)

    output_samples = default_func(input_samples)
    output_mean = output_samples.mean(axis=1)
    output_std = output_samples.std(axis=1)
    
    # Compute first-order and total-order sobol indices using scipy.stats.sobol_indices function
    indices = stats.sobol_indices(func=default_func, n=n_samples, dists=depmod.marginals.values())
    
    # Extract results, and save to dataframe

    input_names = list(marginals.keys())
    output_names = ['strategic_default','cashflow_default','double_trigger_default','any_default']

    input_name_list = []
    output_name_list = []
    input_mean_list = []
    input_std_list = []
    output_mean_list = []
    output_std_list = []
    sobol_first_order_list = []
    sobol_total_order_list = []

    for i in range(len(input_names)):
        for j in range(len(output_names)):

            input_name_list.append(input_names[i])
            input_mean_list.append(input_mean[i])
            input_std_list.append(input_std[i])

            output_name_list.append(output_names[j])
            output_mean_list.append(output_mean[j])
            output_std_list.append(output_std[j])

            sobol_first_order_list.append(indices.first_order[j,i])
            sobol_total_order_list.append(indices.total_order[j,i])

    data = {'loan_id':borrower.loan_id,
            'period':borrower.period,
            'replicate':borrower.replicate,
            'input_name':input_name_list,
            'output_name':output_name_list,
            'input_mean':input_mean_list,
            'input_std':input_std_list,
            'output_mean':output_mean_list,
            'output_std':output_std_list,
            'sobol_first_order':sobol_first_order_list,
            'sobol_total_order':sobol_total_order_list}

    df = pd.DataFrame(data)
    
    return(df,depmod)

### *** INITIAL SETUP *** ###

pwd = os.getcwd()

# Get index of county to run
county_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])

# Read in data on flood-exposed borrowers
damaged_sim_path = '/proj/characklab/flooddata/NC/multiple_events/analysis/mortgage_borrower_simulation_base_case_postprocessed/simulation_output_damaged.parquet'
damaged_sim_df = pd.read_parquet(damaged_sim_path)

# Read in data on kriged property values
counties = np.sort(damaged_sim_df['county'].unique())
county = counties[county_idx]
damaged_sim_df = damaged_sim_df[damaged_sim_df['county']==county].reset_index(drop=True)
county_code = damaged_sim_df['loan_id'].str.split('-')[0][0]

# Create output folder
outfolder = os.path.join(pwd,f'sobol_indices/{county}')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

### *** PROPERTY VALUE DATA *** ###

# Read in output from property value kriging
property_values_path = f'/proj/characklab/flooddata/NC/multiple_events/analysis/property_value_estimates_by_county/{county}/{county}_property_values_kriged.parquet'
property_values = pd.read_parquet(property_values_path)
property_values.rename(columns={'period':'quarter','sigma_log_val_transfer_kriged':'logpv_sigma'},inplace=True)

# Attach info on property value kriging variance to borrower data
damaged_sim_df['quarter'] = damaged_sim_df['period'].apply(lambda x: f'{x.year}Q{x.quarter}')
damaged_sim_df = pd.merge(damaged_sim_df,property_values[['building_id','quarter','logpv_sigma']],on=['building_id','quarter'],how='left')

# Calculate parameters of lognormal distribution used to define kriged property value distribution
damaged_sim_df['logpv_sigma'] = np.maximum(damaged_sim_df['logpv_sigma'],0.01)
damaged_sim_df['logpv_mu'] = np.log(damaged_sim_df['property_value']) - 0.5*damaged_sim_df['logpv_sigma']**2

# Get flood-exposed borrowers who are uninsured 
uninsured_damage_df = damaged_sim_df[damaged_sim_df['uninsured_damage'] > 0].reset_index(drop=True)

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

income_growth = income_df.set_index('period')['per_capita_income']

# Convert to a growth rate (i.e., monthly % change in income)
income_growth = (income_growth.shift(-1)/income_growth).ffill() - 1

### *** DAMAGE COST DATA *** ###

# Read in data on conditional variance in damage cost estimate
# (variance of (y - y_pred) estimated based on residuals from cross-validation as a function of y_pred) 

filepath = os.path.join(pwd,'conditional_variance_models/damage_cost_conditional_variance.pickle')
with open(filepath,'rb') as file:
    damage_cost_conditional_variance = pickle.load(file)
    
### *** COMPUTE SOBOL INDICES FOR EACH FLOOD-EXPOSED BORROWER *** ###

# Specify assumed annual volatilty in income
# (roughly guesstimated from biannual volatility estimated from PSID)
# See bottom left panel of Figure 3 in following paper: doi:10.1515/1935-1682.3347

# Assumed biannual volatility (~10% when looking at middle of income distribution)
biannual_volatility_psid = 0.1

# Convert to annual volatility by dividing by square root of time 
sigma = biannual_volatility_psid / np.sqrt(2)

sobol_list = []

for i in range(len(uninsured_damage_df)):
    
    borrower = uninsured_damage_df.loc[i]
    results,depmod = compute_sobol_indices(borrower,income_growth,damage_cost_conditional_variance,n_samples=2**16,sigma=sigma)
    sobol_list.append(results)
    
if len(sobol_list) > 0:
    sobol_df = pd.concat(sobol_list).reset_index(drop=True)
    outname = os.path.join(outfolder,f'{county}_sobol_indices.parquet')
    sobol_df.to_parquet(outname)