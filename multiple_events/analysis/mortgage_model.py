import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.interpolate as interp
import datetime as dt
import dependence_modeling as dm

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
     
    # Round up to nearest cent
    c = np.ceil(c*100)/100
    
    return(c)

def monthly_interest(r,UPB):
    """
    This function calculates the monthly interest on a fixed-rate loan. 
    
    param: r: annual interest rate on loan (expressed as a percentage from 0-100)
    param: UPB: current unpaid balance on loan
    returns: i: interest accumulated on unpaid balance over the course of one month.  
    """
    # Convert the annual interest rate to a monthly interest rate, and make decimal rather than percent
    r = (r/100)/12
    
    # Calculate dollar amount of interest accumulated over month
    i = UPB*r
    return(i)

def prepayment_hazard(t,S,beta=0,lam=1000,t_cutoff=360):
    """
    This function takes the survival function and coefficient from a fit cox proportional hazards model
    and uses them to construct smooth functions describing the hazard rate and monthly prepayment probability 
    over time and as a function of interest rate spread versus the current market rate. 
    
    param: t: numpy array of timepoints
    param: S: numpy array containing values of survival function at each timepoint. S(t) should correspond to the 
              "baseline survival function" (i.e., S(t) when the interest rate is equal to the market rate for all t). 
    param: beta: coefficient of interest rate spread versus the current market rate from fit cox model
    param: lam: parameter controlling degree of smoothing applied to numerically-calculated hazard rate values
    param: t_cutoff: when creating hazard rate function, assume that h(t > t_cutoff) = h(t_cutoff).
                     This is useful when we have few observations beyond a certain timepoint.
    returns: hazard_rate: function describing instantaneous hazard rate as a function of time and spread versus market rate. 
    returns: monthly_prob: function describing monthly prepayment probability as a function of time and spread versus market rate. 
    """
    # Numerically caculate the baseline hazard rate from the survival function
    # using the following relation: h(t) = -d/dt log(S(t))
    # See https://en.wikipedia.org/wiki/Survival_analysis#Hazard_function_and_cumulative_hazard_function
    y = -1*np.gradient(np.log(S),t)
    
    # Drop entries from beyond cutoff time
    y = y[t <= t_cutoff]
    t = t[t <= t_cutoff]
    
    # Model baseline hazard rate as a smoothed spline
    spl = interp.make_smoothing_spline(t, y, lam=lam)
    h0 = lambda t: np.maximum(spl(np.minimum(t,t_cutoff)),0)
    
    # Create functions return hazard rate and monthly prepayment probability as a function of 
    # time (t) and spread versus current average rate (x). 
    hazard_rate = lambda t,x: h0(t)*np.exp(beta*x)
    monthly_prob = lambda t,x: 1 - np.exp(-1*hazard_rate(t,x))
    
    return(hazard_rate,monthly_prob)

class HomeRepairLoan:
    """
    This class keeps track of the unpaid balance and monthly payment on a home repair loan. 
    """
    
    def __init__(self,loan_amount,loan_term,interest_rate):
        """
        Initialize loan. 
        """
        
        self.interest_rate = interest_rate
        self.unpaid_balance = loan_amount
        self.monthly_payment = monthly_payment(interest_rate,loan_term,loan_amount)
        
    def update_balance(self):
        """
        When called, this method updates the balance of the home repair loan in response to a borrower making a payment.
        """
        # Calculate dollar amount of interest accumulated over month
        interest = monthly_interest(self.interest_rate,self.unpaid_balance)
        
        # Update unpaid balance at end of month to reflect interest + monthly payment
        self.unpaid_balance = max(self.unpaid_balance + interest - self.monthly_payment,0)
        
        # In the event that remaining UPB + interest is less than the next month's payment, update monthly payment
        next_month_interest = monthly_interest(self.interest_rate,self.unpaid_balance)
        self.monthly_payment = np.ceil(min(self.unpaid_balance + next_month_interest,self.monthly_payment)*100)/100

class RecoveryFundingDecisionTree:
    """
    This submodel stochastically allocates different sources of post-disaster recovery funding based on the amount of 
    uninsured damage they sustained as well as their pre-flood financial characteristics. 
    """
    
    def __init__(self,
                 sba_approval_prob,
                 ihp_fraction_of_damage_dist,
                 ihp_fraction_of_limit_dist,
                 max_LTV_private=1.0,
                 max_DTI_private=0.5,
                 months_savings=1):
        
        """
        param: sba_approval_prob: function returning probability of being approved for an SBA loan given credit score and DTI
        param: ihp_fraction_of_damage_dist: distribution of % of damage covered by IHP aid among those with damage <= limit
        param: ihp_fraction_of_limit_dist: distribution of % of IHP aid limit paid out among those with damage > limit
        param: max_LTV_private: highest LTV ratio for which a borrower can be approved for a private home repair loan
        param: max_DTI_private: highest DTI ratio for which a borrower can be approved for a private home repair loan
        param: months_savings: number of months of gross income saved in emergency fund
        """
        
        self.sba_approval_prob = sba_approval_prob
        self.ihp_fraction_of_damage_dist = ihp_fraction_of_damage_dist
        self.ihp_fraction_of_limit_dist = ihp_fraction_of_limit_dist
        self.max_LTV_private = max_LTV_private
        self.max_DTI_private = max_DTI_private
        self.months_savings = months_savings
        
        
    def roll_for_funding(self,
                         funding_gap,
                         monthly_income,
                         property_value,
                         credit_score,
                         unpaid_balance_on_all_loans,
                         total_monthly_debt_obligations,
                         sba_repair_rate,
                         private_repair_rate,
                         max_ihp_grant):
        """
        In response to uninsured damage, borrowers will attempt to access different funding sources for disaster recovery.
        The probability of success will depend on both the amount of damage and their pre-flood financial profile. 
        
        param: funding_gap: dollar amount of funds needed for repairs and recovery
        param: monthly_income: borrower monthly income
        param: propety_value: current value of property
        param: unpaid_balance_on_all_loans: total unpaid balance of loans on property
        param: total_monthly_debt_obligations: total monthly debt obligations of borrower
        param: sba_repair_rate: interest rate on SBA home repair loans
        param: private_repair_rate: interest rate on private home repair loans
        param: max_ihp_grant: maximum FEMA IHP assistance grant amount
        """
        
        sba_loan_amount = 0
        ihp_grant_amount = 0
        private_loan_amount = 0
        savings_amount = 0
                
        # Compute the probability of being approved for an SBA loan based on credit score and DTI
        sba_payment = monthly_payment(sba_repair_rate,360,funding_gap)
        sba_DTI = (total_monthly_debt_obligations + sba_payment) / monthly_income
        sba_prob = self.sba_approval_prob(credit_score,sba_DTI)
        
        # Roll for SBA loan approval
        
        if np.random.binomial(1,sba_prob):
            
            sba_loan_amount = funding_gap
            funding_gap = 0
            
        else:
            
            # If denied an SBA loan, roll for FEMA IHP
            
            if funding_gap <= max_ihp_grant:
                ihp_grant_amount = funding_gap*self.ihp_fraction_of_damage_dist.rvs()[0]
            else:
                ihp_grant_amount = max_ihp_grant*self.ihp_fraction_of_limit_dist.rvs()[0]
                
            # Round up to nearest cent
            ihp_grant_amount = np.ceil(ihp_grant_amount*100)/100
            
            # Subtract FEMA home repair grant amount from funding gap
            funding_gap = max(funding_gap - ihp_grant_amount,0)
            
            # If funding gap remains, attempt to secure private loan, borrowing against equity in home
            if funding_gap > 0:
                
                private_payment = monthly_payment(private_repair_rate,360,funding_gap)
                private_DTI = (total_monthly_debt_obligations + private_payment) / monthly_income
                private_LTV = (unpaid_balance_on_all_loans + funding_gap) / property_value
                
                if (private_DTI <= self.max_DTI_private) and (private_LTV <= self.max_LTV_private):
                    
                    private_loan_amount = funding_gap
                    funding_gap = 0
                    
                else:
                    # If not approved for private loan, turn to savings as a last resort
                    emergency_fund = self.months_savings*monthly_income
                    savings_amount = min(emergency_fund,funding_gap)
                    funding_gap -= savings_amount
        
        return(sba_loan_amount,ihp_grant_amount,private_loan_amount,savings_amount,funding_gap)
        
        
class MortgageBorrower:
    
    """
    This class simulates the financial conditions of a residential mortgage borrower subject to flood damage exposure. 
    """
    
    def __init__(self,loan_id,building_id,origination_period,loan_purpose,loan_amount,loan_term,interest_rate,income,DTI,credit_score):
        """
        param: loan_id: unique identifier used to distinguish between mortgage loans 
        param: building_id: unique identifier tying mortgage to a specific property
        param: origination_period: year and month in which the mortgage was originated
        param: loan_purpose: string denoting whether mortgage was for home purchase or refinancing
        param: loan_amount: initial unpaid balance on loan
        param: loan_term: number of months until the loan reaches maturity
        param: interest_rate: interest_rate on loan
        param: income: annual income of borrower at origination
        param: DTI: debt-to-income ratio of borrower at origination
        param: credit_score: borrower credit score on 300-850 scale. 
        """
        self.loan_id = loan_id
        self.building_id = building_id
        self.origination_period = origination_period
        self.credit_score = credit_score
        self.loan_purpose = loan_purpose
        self.loan_amount = loan_amount
        self.loan_term = loan_term
        self.interest_rate = interest_rate
        self.monthly_mortgage_payment = monthly_payment(interest_rate,loan_term,loan_amount)
        self.monthly_income_at_origination = income/12
        self.other_monthly_debt_payments = self.monthly_income_at_origination*DTI - self.monthly_mortgage_payment
        
    def initialize_state_variables(self,property_value,market_rate,private_repair_rate,sba_repair_rate,ihp_limit,income_growth,property_damage_exposure,RFDT,end_period=None):
        """
        Create time-dependent state variables describing borrower financial conditions.  
        
        param: property_value: pandas series of property value estimates indexed by period
        param: market_rate: pandas series of market mortgage rate estimates indexed by period
        param: private_repair_rate: pandas series of interest rates on private home repair loans indexed by period
        param: sba_repair_rate: pandas series of interest rates on SBA home repair loans indexed by period
        param: ihp_limit: maximum FEMA home repair grant amount indexed by period
        param: income_growth: pandas series of monthly income growth rate indexed by period
        param: property_damage_exposure: pandas dataframe of flood damage costs indexed by period
        param: RFDT: Instance of RecoveryFundingDecisionTree class. Used to allocate sources of disaster recovery funding. 
        param: end_period: period beyond which to stop simulation. if none, defaults to period at end of loan term. 
        """
        
        self.RFDT = RFDT
        
        if end_period is None:
            end_period = self.origination_period + pd.offsets.MonthEnd(self.loan_term-1)
        
        self.periods = pd.period_range(self.origination_period,end_period)
        n_periods = len(self.periods)
        
        self.loan_age = pd.Series(np.arange(n_periods),index=self.periods)
        self.property_value = property_value[self.periods]
        self.market_rate = market_rate[self.periods]
        self.rate_spread = self.interest_rate - self.market_rate
        self.private_repair_rate = private_repair_rate[self.periods]
        self.sba_repair_rate = sba_repair_rate[self.periods]
        self.ihp_limit = ihp_limit[self.periods]
        self.insured_damage = property_damage_exposure['insured_nominal_cost'][self.periods]
        self.uninsured_damage = property_damage_exposure['uninsured_nominal_cost'][self.periods]
        
        income_growth = income_growth[self.periods]
        self.monthly_income = self.monthly_income_at_origination*((1 + income_growth).shift().cumprod().fillna(1))
        
        self.unpaid_mortgage_balance = pd.Series(np.ones(n_periods)*np.nan,index=self.periods)
        self.unpaid_mortgage_balance[self.origination_period] = self.loan_amount
        
        self.unpaid_repair_loan_balance = pd.Series(np.zeros(n_periods),index=self.periods)
        self.monthly_repair_loan_payments = pd.Series(np.zeros(n_periods),index=self.periods)
        self.number_of_repair_loans = pd.Series(np.zeros(n_periods),index=self.periods)
        
        self.unpaid_balance_on_all_loans = pd.Series(np.ones(n_periods)*np.nan,index=self.periods)
        self.total_monthly_debt_obligations = pd.Series(np.ones(n_periods)*np.nan,index=self.periods)
                
        self.home_equity = pd.Series(np.ones(n_periods)*np.nan,index=self.periods)
        self.total_mortgage_payments = pd.Series(np.zeros(n_periods),index=self.periods)
        
        self.LTV = pd.Series(np.ones(n_periods)*np.nan,index=self.periods)
        self.LTV_excluding_repair_loans = pd.Series(np.ones(n_periods)*np.nan,index=self.periods)
        
        self.DTI = pd.Series(np.ones(n_periods)*np.nan,index=self.periods)
        self.DTI_excluding_repair_loans = pd.Series(np.ones(n_periods)*np.nan,index=self.periods)
        
        self.sba_loan_recovery_amount = pd.Series(np.zeros(n_periods),index=self.periods)
        self.ihp_grant_recovery_amount = pd.Series(np.zeros(n_periods),index=self.periods)
        self.private_loan_recovery_amount = pd.Series(np.zeros(n_periods),index=self.periods)
        self.savings_recovery_amount = pd.Series(np.zeros(n_periods),index=self.periods)
        self.recovery_funding_gap = pd.Series(np.zeros(n_periods),index=self.periods)
        
        self.termination_code = pd.Series(n_periods*[pd.NA],dtype='string',index=self.periods)
        
    def simulate_repayment(self,monthly_prepayment_prob,LTV_cutoff=1.0):
        """
        Simulate borrower financial conditions over time until mortgage loan is repaid. 
        
        param: monthly_prepayment_prob: function returning monthly prepayment probability given loan age and rate spread
        param: LTV_cutoff: maximum allowable LTV for a borrower to sell or refinance their home 
        """
        
        self.repair_loans = []
        
        # Period corresponding to end of simulation time horizon
        end_period = self.periods[-1]
        
        for period in self.periods:
                        
            # Get unpaid balance and monthly payments on any prexisting home repair loans
            for i in range(len(self.repair_loans)):
                self.unpaid_repair_loan_balance[period] += self.repair_loans[i].unpaid_balance
                self.monthly_repair_loan_payments[period] += self.repair_loans[i].monthly_payment
                self.number_of_repair_loans[period] += (self.repair_loans[i].unpaid_balance > 0).astype(int)
                
            # Get total unpaid balance on all home equity-based loans
            self.unpaid_balance_on_all_loans[period] = self.unpaid_mortgage_balance[period] + self.unpaid_repair_loan_balance[period]
            
            # Get total monthly debt obligations
            self.total_monthly_debt_obligations[period] = self.monthly_mortgage_payment + self.monthly_repair_loan_payments[period] + self.other_monthly_debt_payments
            
            # Calculate home equity and LTV
            self.home_equity[period] = self.property_value[period] - self.unpaid_balance_on_all_loans[period]
            self.LTV[period] = self.unpaid_balance_on_all_loans[period] / self.property_value[period]
            self.LTV_excluding_repair_loans[period] = self.unpaid_mortgage_balance[period] / self.property_value[period]
            
            # Calculate DTI
            self.DTI[period] = self.total_monthly_debt_obligations[period] / self.monthly_income[period]
            self.DTI_excluding_repair_loans[period] = (self.monthly_mortgage_payment + self.other_monthly_debt_payments) / self.monthly_income[period]
            
            
            # If LTV is below cutoff, roll for probability of prepayment
            if (self.LTV[period] <= LTV_cutoff):
                p = monthly_prepayment_prob(self.loan_age[period],self.rate_spread[period])
                prepay = np.random.binomial(1,p)
            else:
                prepay = 0
                
            # If exposed to uninsured property damage during month, create a new home repair loan
            
            default = 0 # Only modeling disaster-related defaults currently
            
            if self.uninsured_damage[period] > 0:
                
                # De-prioritize prepayment
                prepay = 0
                
                # Roll for sources of post-disaster recovery funding
                funding_sources = self.RFDT.roll_for_funding(self.uninsured_damage[period],
                                                             self.monthly_income[period],
                                                             self.property_value[period],
                                                             self.credit_score,
                                                             self.unpaid_balance_on_all_loans[period],
                                                             self.total_monthly_debt_obligations[period],
                                                             self.sba_repair_rate[period],
                                                             self.private_repair_rate[period],
                                                             self.ihp_limit[period])
                
                sba_loan_amount,ihp_grant_amount,private_loan_amount,savings_amount,funding_gap = funding_sources
                
                self.sba_loan_recovery_amount[period] = sba_loan_amount
                self.ihp_grant_recovery_amount[period] = ihp_grant_amount
                self.private_loan_recovery_amount[period] = private_loan_amount
                self.savings_recovery_amount[period] = savings_amount
                self.recovery_funding_gap[period] = funding_gap
                
                # If a funding gap remains, assume the homeowner defaults
                
                if funding_gap > 0:
                    
                    default = 1
                    
                else:
                                        
                    # Add SBA and private home repair loans to debt obligations 
                    if sba_loan_amount > 0:
                        self.repair_loans.append(HomeRepairLoan(sba_loan_amount,360,self.sba_repair_rate[period]))
                    if private_loan_amount > 0:
                        self.repair_loans.append(HomeRepairLoan(private_loan_amount,360,self.private_repair_rate[period]))
            
            # At end of month, update loan balances
            next_period = period + pd.offsets.MonthEnd(1)
            
            for i in range(len(self.repair_loans)):
                self.repair_loans[i].update_balance()
            
            # Calculate dollar amount of interest accumulated on mortgage over month
            interest = monthly_interest(self.interest_rate,self.unpaid_mortgage_balance[period])
            
            # Calculate amount paid towards mortgage in month (including prepayment)
            payment = max(prepay*(self.unpaid_mortgage_balance[period] + interest),self.monthly_mortgage_payment)
            payment = np.ceil(payment*100)/100 # Round up to nearest cent
            
            self.total_mortgage_payments[period] = payment
        
            # Update unpaid balance at end of month to reflect interest + monthly payment
            next_period_balance = max(self.unpaid_mortgage_balance[period] + interest - payment,0)
            
            if (next_period_balance > 0) and (period != end_period) and not default:
                self.unpaid_mortgage_balance[next_period] = next_period_balance
            elif (next_period_balance == 0) or default:
                self.termination_date = period
                
                if default:
                    self.termination_code[period] = 'D'
                else:
                    self.termination_code[period] = 'P'
                    
                break
            else:
                self.termination_date = pd.NaT
                break
            
    def summarize(self):
        """
        Return a monthly summary of borrower financial conditions
        """
            
        if self.termination_date < self.periods[-1]:
            
            self.periods = self.periods[self.periods <= self.termination_date]
            
        summary = pd.DataFrame(data={'period':self.periods})
        summary['date'] = summary['period'].dt.start_time
        
        # Add time-invariant borrower characteristics 
        summary['loan_id'] = self.loan_id
        summary['building_id'] = self.building_id
        summary['credit_score'] = self.credit_score
        summary['loan_purpose'] = self.loan_purpose
        summary['loan_term'] = self.loan_term
        summary['interest_rate'] = self.interest_rate
        
        # Index by period
        summary.set_index('period',inplace=True)
        
        # Add time-varying borrower characteristics
        summary['market_rate'] = self.market_rate[self.periods]
        summary['loan_age'] = self.loan_age[self.periods]
        summary['insured_damage'] = self.insured_damage[self.periods]
        summary['uninsured_damage'] = self.uninsured_damage[self.periods]
        summary['number_of_repair_loans'] = self.number_of_repair_loans[self.periods]
        summary['unpaid_mortgage_balance'] = self.unpaid_mortgage_balance[self.periods]
        summary['unpaid_repair_loan_balance'] = self.unpaid_repair_loan_balance[self.periods]
        summary['unpaid_balance_on_all_loans'] = self.unpaid_balance_on_all_loans[self.periods]
        summary['property_value'] = self.property_value[self.periods]
        summary['home_equity'] = self.home_equity[self.periods]
        summary['mortgage_payments'] = self.total_mortgage_payments[self.periods]
        summary['repair_loan_payments'] = self.monthly_repair_loan_payments[self.periods]
        summary['other_debt_payments'] = self.other_monthly_debt_payments
        summary['monthly_debt_obligations'] = self.total_monthly_debt_obligations[self.periods]
        summary['monthly_income'] = self.monthly_income[self.periods]
        summary['LTV'] = self.LTV[self.periods]
        summary['DTI'] = self.DTI[self.periods]
        summary['LTV_excluding_repair_loans'] = self.LTV_excluding_repair_loans[self.periods]
        summary['DTI_excluding_repair_loans'] = self.DTI_excluding_repair_loans[self.periods]
        
        summary['sba_loan_recovery_amount'] = self.sba_loan_recovery_amount[self.periods]
        summary['ihp_grant_recovery_amount'] = self.ihp_grant_recovery_amount[self.periods]
        summary['private_loan_recovery_amount'] = self.private_loan_recovery_amount[self.periods]
        summary['savings_recovery_amount'] = self.savings_recovery_amount[self.periods]
        summary['recovery_funding_gap'] = self.recovery_funding_gap[self.periods]
        
        summary['termination_code'] = self.termination_code[self.periods]
        
        summary.reset_index(inplace=True)
        
        return(summary)