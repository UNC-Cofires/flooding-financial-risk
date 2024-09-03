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

class FixedRateLoan:
    """
    This class keeps track of the unpaid balance and monthly payment on a fixed-rate loan. 
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
        When called, this method updates the balance of the loan in response to a borrower making a payment.
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
        self.loan_term = loan_term
        self.mortgage_loan = FixedRateLoan(loan_amount,loan_term,interest_rate)
        self.monthly_income_at_origination = income/12
        self.other_monthly_debt_payments = self.monthly_income_at_origination*DTI - self.mortgage_loan.monthly_payment
        
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
        
        self.loan_age = np.arange(n_periods)
        self.property_value = property_value[self.periods].to_numpy()
        self.market_rate = market_rate[self.periods].to_numpy()
        self.repair_rate = repair_rate[self.periods].to_numpy()
        self.rate_spread = self.mortgage_loan.interest_rate - self.market_rate
        self.insured_damage = property_damage_exposure['insured_nominal_cost'][self.periods].to_numpy()
        self.uninsured_damage = property_damage_exposure['uninsured_nominal_cost'][self.periods].to_numpy()
        
        income_growth = income_growth[self.periods]
        self.monthly_income = self.monthly_income_at_origination*((1 + income_growth).shift().cumprod().fillna(1)).to_numpy()
        
        self.unpaid_mortgage_balance = np.zeros(n_periods)
        self.unpaid_mortgage_balance[0] = self.mortgage_loan.unpaid_balance
        
        self.unpaid_repair_loan_balance = np.zeros(n_periods)
        self.monthly_repair_loan_payments = np.zeros(n_periods)
        self.number_of_repair_loans = np.zeros(n_periods)
        
        self.unpaid_balance_on_all_loans = np.zeros(n_periods)
        self.total_monthly_debt_obligations = np.zeros(n_periods)
                
        self.home_equity = np.zeros(n_periods)
        self.total_mortgage_payments = np.zeros(n_periods)
        
        self.LTV = np.zeros(n_periods)
        self.LTV_excluding_repair_loans = np.zeros(n_periods)
        self.aLTV = np.ones(n_periods)*np.nan
        
        self.DTI = np.zeros(n_periods)
        self.DTI_excluding_repair_loans = np.zeros(n_periods)
        self.aDTI = np.ones(n_periods)*np.nan
        
        self.termination_code = pd.Series(n_periods*[pd.NA],dtype='string')
        
    def simulate_repayment(self,monthly_prepayment_prob,LTV_cutoff=1.0,aLTV_cutoff=1.0,aDTI_cutoff=0.45):
        """
        Simulate borrower financial conditions over time until mortgage loan is repaid. 
        
        param: monthly_prepayment_prob: function returning monthly prepayment probability given loan age and rate spread
        param: LTV_cutoff: maximum allowable LTV for a borrower to sell or refinance their home
        param: aLTV_cutoff: if aLTV exceeds this threshold, assume the borrower cannot easily finance home repairs, and terminate the simulation.
        param: aDTI_cutoff: if aDTI exceeds this threshold, assume the borrower cannot easily finance home repairs, and terminate the simulation.
        """
        
        self.termination_date = pd.NaT
        self.repair_loans = []
        
        for t in range(len(self.periods)):
            
            # Get unpaid balance on mortgage
            self.unpaid_mortgage_balance[t] = self.mortgage_loan.unpaid_balance
                                    
            # Get unpaid balance and monthly payments on any prexisting home repair loans
            for i in range(len(self.repair_loans)):
                self.unpaid_repair_loan_balance[t] += self.repair_loans[i].unpaid_balance
                self.monthly_repair_loan_payments[t] += self.repair_loans[i].monthly_payment
                self.number_of_repair_loans[t] += (self.repair_loans[i].unpaid_balance > 0).astype(int)
                
            # Get total unpaid balance on all home equity-based loans
            self.unpaid_balance_on_all_loans[t] = self.mortgage_loan.unpaid_balance + self.unpaid_repair_loan_balance[t]
            
            # Get total monthly debt obligations
            self.total_monthly_debt_obligations[t] = self.mortgage_loan.monthly_payment + self.monthly_repair_loan_payments[t] + self.other_monthly_debt_payments
            
            # Calculate home equity and LTV
            self.home_equity[t] = self.property_value[t] - self.unpaid_balance_on_all_loans[t]
            self.LTV[t] = self.unpaid_balance_on_all_loans[t] / self.property_value[t]
            self.LTV_excluding_repair_loans[t] = self.mortgage_loan.unpaid_balance / self.property_value[t]
            
            # Calculate DTI
            self.DTI[t] = self.total_monthly_debt_obligations[t] / self.monthly_income[t]
            self.DTI_excluding_repair_loans[t] = (self.mortgage_loan.monthly_payment + self.other_monthly_debt_payments) / self.monthly_income[t]
            
            # If LTV is below cutoff, roll for probability of prepayment
            if self.LTV[t] <= LTV_cutoff:
                p = monthly_prepayment_prob(self.loan_age[t],self.rate_spread[t])
                prepay = np.random.binomial(1,p)
            else:
                prepay = 0
                
            # Assess response to uninsured damage
            terminate = 0
            if self.uninsured_damage[t] > 0:
                
                # Prioritize home repairs over prepayment
                prepay = 0
                
                # Calculate damage-adjusted LTV and DTI ratios based on a hypothetical home repair loan
                repair_loan = FixedRateLoan(self.uninsured_damage[t],360,self.repair_rate[t])
                self.aLTV[t] = (self.unpaid_balance_on_all_loans[t] + repair_loan.unpaid_balance) / self.property_value[t]
                self.aDTI[t] = (self.total_monthly_debt_obligations[t] + repair_loan.monthly_payment) / self.monthly_income[t]
                
                # Assume borrower is approved if they can support this loan without exceeding LTV and DTI cutoffs
                # Otherwise, assume they can't easily finance repairs, and terminate the simulation
                if (self.aLTV[t] <= aLTV_cutoff) and (self.aDTI[t] <= aDTI_cutoff):
                    self.repair_loans.append(repair_loan)
                else:
                    terminate=1
            
            # At end of month, update balance of mortgage and repair loans
            self.total_mortgage_payments[t] = prepay*self.mortgage_loan.unpaid_balance + (1-prepay)*self.mortgage_loan.monthly_payment
            
            self.mortgage_loan.update_balance()
            
            for i in range(len(self.repair_loans)):
                self.repair_loans[i].update_balance()
                
            # Stop simulation if loan is prepaid or otherwise removed
            if prepay:
                self.termination_code[t] = 'P'
                self.termination_date = self.periods[t]
                break
            elif terminate:
                self.termination_code[t] = 'T'
                self.termination_date = self.periods[t]
                break
            
    def summarize(self):
        """
        Return a monthly summary of borrower financial conditions
        """
        # (!) Need to fix this stuff     
        if self.termination_date < self.periods[-1]:
            self.periods = self.periods[self.periods <= self.termination_date]
            
        n_periods = len(self.periods)
            
        summary = pd.DataFrame(data={'period':self.periods})
        summary['date'] = summary['period'].dt.start_time
        
        # Add time-invariant borrower characteristics 
        summary['loan_id'] = self.loan_id
        summary['building_id'] = self.building_id
        summary['credit_score'] = self.credit_score
        summary['loan_purpose'] = self.loan_purpose
        summary['loan_term'] = self.loan_term
        summary['interest_rate'] = self.mortgage_loan.interest_rate
        
        # Add time-varying borrower characteristics
        summary['loan_age'] = self.loan_age[:n_periods]
        summary['insured_damage'] = self.insured_damage[:n_periods]
        summary['uninsured_damage'] = self.uninsured_damage[:n_periods]
        summary['number_of_repair_loans'] = self.number_of_repair_loans[:n_periods]
        summary['market_rate'] = self.market_rate[:n_periods]
        summary['repair_rate'] = self.repair_rate[:n_periods]
        summary['unpaid_mortgage_balance'] = self.unpaid_mortgage_balance[:n_periods]
        summary['unpaid_repair_loan_balance'] = self.unpaid_repair_loan_balance[:n_periods]
        summary['unpaid_balance_on_all_loans'] = self.unpaid_balance_on_all_loans[:n_periods]
        summary['property_value'] = self.property_value[:n_periods]
        summary['home_equity'] = self.home_equity[:n_periods]
        summary['mortgage_payments'] = self.total_mortgage_payments[:n_periods]
        summary['repair_loan_payments'] = self.monthly_repair_loan_payments[:n_periods]
        summary['other_debt_payments'] = self.other_monthly_debt_payments
        summary['monthly_debt_obligations'] = self.total_monthly_debt_obligations[:n_periods]
        summary['monthly_income'] = self.monthly_income[:n_periods]
        summary['LTV'] = self.LTV[:n_periods]
        summary['DTI'] = self.DTI[:n_periods]
        summary['LTV_excluding_repair_loans'] = self.LTV_excluding_repair_loans[:n_periods]
        summary['DTI_excluding_repair_loans'] = self.DTI_excluding_repair_loans[:n_periods]
        summary['aLTV'] = self.aLTV[:n_periods]
        summary['aDTI'] = self.aDTI[:n_periods]

        summary['termination_code'] = self.termination_code[:n_periods]
                
        return(summary)