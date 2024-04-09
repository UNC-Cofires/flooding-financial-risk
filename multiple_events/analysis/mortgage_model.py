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
        param: credit_score: credit score of the borrower at origination
        """
        self.loan_id = loan_id
        self.building_id = building_id
        self.credit_score = credit_score
        self.origination_period = origination_period
        self.loan_purpose = loan_purpose
        self.loan_amount = loan_amount
        self.loan_term = loan_term
        self.interest_rate = interest_rate
        self.monthly_mortgage_payment = monthly_payment(interest_rate,loan_term,loan_amount)
        self.monthly_income_at_origination = income/12
        self.other_monthly_debt_payments = self.monthly_income_at_origination*DTI - self.monthly_mortgage_payment
        
    def initialize_state_variables(self,property_value,market_rate,repair_rate,income_growth,property_damage_exposure,end_period=None):
        """
        Create time-dependent state variables describing borrower financial conditions.  
        
        param: property_value: pandas series of property value estimates indexed by period
        param: market_rate: pandas series of market mortgage rate estimates indexed by period
        param: repair_rate: pandas series of interest rates on home repair / disaster recovery loans indexed by period
        param: income_growth: pandas series of monthly income growth rate indexed by period
        param: property_damage_exposure: pandas dataframe of flood damage costs indexed by period
        param: end_period: period beyond which to stop simulation. if none, defaults to period at end of loan term. 
        """
        
        if end_period is None:
            end_period = self.origination_period + pd.offsets.MonthEnd(self.loan_term-1)
        
        self.periods = pd.period_range(self.origination_period,end_period)
        n_periods = len(self.periods)
        
        self.loan_age = pd.Series(np.arange(n_periods),index=self.periods)
        self.property_value = property_value[self.periods]
        self.market_rate = market_rate[self.periods]
        self.repair_rate = repair_rate[self.periods]
        self.rate_spread = self.interest_rate - self.market_rate
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
        self.aLTV = pd.Series(np.ones(n_periods)*np.nan,index=self.periods)
        
        self.DTI = pd.Series(np.ones(n_periods)*np.nan,index=self.periods)
        self.DTI_excluding_repair_loans = pd.Series(np.ones(n_periods)*np.nan,index=self.periods)
        self.aDTI = pd.Series(np.ones(n_periods)*np.nan,index=self.periods)
        
        self.termination_code = pd.Series(n_periods*[pd.NA],dtype='string',index=self.periods)
        
    def simulate_repayment(self,monthly_prepayment_prob,LTV_cutoff=1.0,aLTV_cutoff=1.0,aDTI_cutoff=0.5):
        """
        Simulate borrower financial conditions over time until mortgage loan is repaid. 
        
        param: monthly_prepayment_prob: function returning monthly prepayment probability given loan age and rate spread
        param: LTV_cutoff: maximum allowable LTV for a borrower to sell or refinance their home
        param: aLTV_cutoff: if aLTV exceeds this threshold, assume the borrower cannot easily finance home repairs, and terminate the simulation.
        param: aDTI_cutoff: if aDTI exceeds this threshold, assume the borrower cannot easily finance home repairs, and terminate the simulation.
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
            if self.LTV[period] <= LTV_cutoff:
                p = monthly_prepayment_prob(self.loan_age[period],self.rate_spread[period])
                prepay = np.random.binomial(1,p)
            else:
                prepay = 0
                
            # Assess response to uninsured damage
            terminate = 0
            if self.uninsured_damage[period] > 0:
                
                # Prioritize home repairs over prepayment
                prepay = 0
                
                # Calculate damage-adjusted LTV and DTI ratios based on a hypothetical home repair loan
                repair_loan = HomeRepairLoan(self.uninsured_damage[period],360,self.repair_rate[period])
                self.aLTV[period] = (self.unpaid_balance_on_all_loans[period] + repair_loan.unpaid_balance) / self.property_value[period]
                self.aDTI[period] = (self.total_monthly_debt_obligations[period] + repair_loan.monthly_payment) / self.monthly_income[period]
                
                # Assume borrower is approved if they can support this loan without exceeding LTV and DTI cutoffs
                # Otherwise, assume they can't easily finance repairs, and terminate the simulation
                if (self.aLTV[period] <= aLTV_cutoff) and (self.aDTI[period] <= aDTI_cutoff):
                    self.repair_loans.append(repair_loan)
                else:
                    terminate=1
            
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
            
            if (next_period_balance > 0) and (period != end_period) and not terminate:
                self.unpaid_mortgage_balance[next_period] = next_period_balance
            elif (next_period_balance == 0) or terminate:
                self.termination_date = period
                
                if terminate:
                    self.termination_code[period] = 'T'
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
        summary['loan_age'] = self.loan_age[self.periods]
        summary['insured_damage'] = self.insured_damage[self.periods]
        summary['uninsured_damage'] = self.uninsured_damage[self.periods]
        summary['number_of_repair_loans'] = self.number_of_repair_loans[self.periods]
        summary['market_rate'] = self.market_rate[self.periods]
        summary['repair_rate'] = self.repair_rate[self.periods]
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
        summary['aLTV'] = self.aLTV[self.periods]
        summary['aDTI'] = self.aDTI[self.periods]

        summary['termination_code'] = self.termination_code[self.periods]
        
        summary.reset_index(inplace=True)
        
        return(summary)