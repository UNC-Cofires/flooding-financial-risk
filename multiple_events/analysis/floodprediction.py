import pandas as pd
import numpy as np
import geopandas as gpd
import datetime as dt
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

# *** Class for performing Random Forest regression on arbitrary data ***

class RegressionObject:

    def __init__(self,train_df,test_df,target_df,response_variable,features):
        """
        param: train_df: pandas dataframe of training data (m x n+1)
        param: test_df: pandas dataframe of validation data (m x n+1)
        param: target_df: pandas dataframe of target data (z x n)
        param: response_variable: name of response variable
        param: features: list of predictors
        """
        self.features = [f for f in features if f != response_variable and f != train_df.index.name]
        self.response_variable = response_variable
        self.x = train_df[self.features].to_numpy()
        self.y = train_df[response_variable].to_numpy()
        self.x_test = test_df[self.features].to_numpy()
        self.y_test = test_df[self.response_variable].to_numpy()
        self.x_target = target_df[self.features].to_numpy()
        self.model = None

        return(None)

    def model_fit(self):
        """
        Fit a random forest regression model to the data
        """
        # R parameters:
        # mtry <- 6
        # min.node.size <- 728
        # sample.fraction <- 0.6295282

        self.model = RandomForestRegressor(max_depth=6)
        self.model.fit(self.x,self.y)

        return(None)

    def model_predict(self,x):
        """
        param: x: testing data
        returns: y_pred: predicted value of response variable
        """
        y_pred = self.model.predict(x)
        return(y_pred)

    def model_classify(self,x,threshold):
        """
        param: x: testing data
        param: threshold: assign 1 if Pr(y=1 | x) > threshold
        """
        y_pred = self.model_predict(x)
        y_class = (y_pred > threshold).astype(int)
        return(y_class)

    def update_test_data(self,test_df):
        """
        param: test_df: new version of testing data to use
        """
        self.x_test = test_df[self.features].to_numpy()
        self.y_test = test_df[self.response_variable].to_numpy()

        return(None)

# Helper function to compute elements of confusion matrix
# as well as optimal probability threshold for classification

def confusion_matrix(y_pred,y_true,threshold):
    """
    Compute the elements of the confusion matrix as well as
    sensitivity, specificity, and precision.

    param: y_pred: numpy array of predicted class probabilities
    param: y_true: numpy array of true class labels
    param: threshold: threshold (cut-point) probability for classification
    """
    y_class = (y_pred > threshold).astype(int)

    TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_class).ravel()

    P = TP + FN
    N = TN + FP

    TPR = TP/P
    TNR = TN/N
    PPV = TP/(TP + FP)

    # Return result as dictionary
    d = {'TP':TP,'FP':FP,'TN':TN,'FN':FN,'TPR':TPR,'TNR':TNR,'PPV':PPV}
    return(d)

def minimized_difference_threshold(y_pred,y_true):
    """
    Calculate the optimal threshold (cut-point) for classificatin based on
    the minimized difference criterion defined by Jimenez-Valverde et al.
    (doi:10.1016/j.actao.2007.02.001)

    param: y_pred: numpy array of predicted class probabilities
    param: y_true: numpy array of true class labels
    """
    TPR = np.vectorize(lambda x: confusion_matrix(y_pred,y_true,x)['TPR'])
    TNR = np.vectorize(lambda x: confusion_matrix(y_pred,y_true,x)['TNR'])
    abs_diff = lambda x: np.abs(TPR(x) - TNR(x))
    threshold = y_pred[np.argmin(abs_diff(y_pred))]

    return(threshold)

# Class for performing K-fold validation

class ValidationObject:

    def __init__(self,data,response_variable,features,k=5,sample_weights=None):

        """
        param: data: pandas dataframe of training data (m x n+1)
        param: response_variable: name of response variable
        param: features: list of predictors
        param: k: number of k-fold cross validation iterations to perform
        param: sample_weights: numpy array of weights for calculation of weighted performance metrics
        """
        self.data = data
        self.features = [f for f in features if f != response_variable and f != data.index.name]
        self.response_variable = response_variable
        self.k = k
        self.kf = KFold(n_splits=self.k,random_state=None,shuffle=True)
        self.sample_weights = sample_weights

        self.ROC_curve = None
        self.PR_curve = None

    def cross_validate(self):
        """
        Perform k-fold cross validation
        """

        results_list = []

        # Split in to k test/train folds
        for k,(train_indices,test_indices) in enumerate(self.kf.split(self.data)):

            results_dict = {'fold':k}

            train_df = self.data.iloc[train_indices].copy()
            test_df = self.data.iloc[test_indices].copy()

            # Fit model
            mod = RegressionObject(train_df,test_df,test_df,self.response_variable,self.features)
            mod.model_fit()

            # Predict probabilities
            y_pred = mod.model_predict(mod.x_test)

            # Compute optimal classification threshold
            y_true = mod.y_test
            threshold = minimized_difference_threshold(y_pred,y_true)
            results_dict['threshold'] = threshold

            # Assign class labels to predictions
            y_class = mod.model_classify(mod.x_test,threshold)

            ## Compute unweighted performance metrics

            # Threshold-independent metrics
            results_dict['roc_auc'] = metrics.roc_auc_score(y_true,y_pred)
            results_dict['avg_prec'] = metrics.average_precision_score(y_true,y_pred)
            results_dict['bs_loss'] = metrics.brier_score_loss(y_true,y_pred)
            results_dict['log_loss'] = metrics.log_loss(y_true,y_pred)

            # Threshold-dependent metrics
            results_dict['accuracy'] = metrics.accuracy_score(y_true,y_class)
            results_dict['balanced_accuracy'] = metrics.balanced_accuracy_score(y_true,y_class)
            results_dict['f1_score'] = metrics.f1_score(y_true,y_class)
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_class).ravel()
            results_dict['sensitivity'] = tp/(tp + fn)
            results_dict['specificity'] = tn/(tn + fp)
            results_dict['precision'] = tp/(tp + fp)

            ## Compute weighted performance metrics

            if self.sample_weights is not None:

                sample_weights = self.sample_weights[test_indices]

                # Threshold-independent metrics
                results_dict['weighted_roc_auc'] = metrics.roc_auc_score(y_true,y_pred,sample_weight=sample_weights)
                results_dict['weighted_avg_prec'] = metrics.average_precision_score(y_true,y_pred,sample_weight=sample_weights)
                results_dict['weighted_bs_loss'] = metrics.brier_score_loss(y_true,y_pred,sample_weight=sample_weights)
                results_dict['weighted_log_loss'] = metrics.log_loss(y_true,y_pred,sample_weight=sample_weights)

                # Threshold-dependent metrics
                results_dict['weighted_accuracy'] = metrics.accuracy_score(y_true,y_class,sample_weight=sample_weights)
                results_dict['weighted_balanced_accuracy'] = metrics.balanced_accuracy_score(y_true,y_class,sample_weight=sample_weights)
                results_dict['weighted_f1_score'] = metrics.f1_score(y_true,y_class,sample_weight=sample_weights)
                tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_class,sample_weight=sample_weights).ravel()
                results_dict['weighted_sensitivity'] = tp/(tp + fn)
                results_dict['weighted_specificity'] = tn/(tn + fp)
                results_dict['weighted_precision'] = tp/(tp + fp)

            results_list.append(results_dict.copy())

        return(pd.DataFrame(results_list))

# *** Flood event class for implementing data processing and prediction workflow

class FloodEvent:

    def __init__(self,study_area,start_date,end_date,peak_date,crs=None):
        """
        Initialize FloodEvent class.

        param: study_area: geodataframe of areas included in study
        param: start_date: start date of event (YYYY-MM-DD)
        param: end_date: end date of event (YYYY-MM-DD)
        param: peak_date: peak date of event (YYYY-MM-DD)
        param (optional): crs: coordinate reference system to use in analysis
        """

        # Specify CRS (or set to that of study area by default)
        if crs is None:
            self.crs = study_area.crs
        else:
            self.crs = crs

        # Dissolve study area so that it's a single (multi)polygon
        if study_area.crs != self.crs:
            study_area = study_area.to_crs(self.crs)

        self.study_area = study_area.dissolve()['geometry'].values[0]

        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.peak_date = pd.Timestamp(peak_date)

        return(None)

    def preprocess_data(self,buildings,claims,policies):
        """
        Determine presence or absence of flooding at building locations based on property-level data

        param: buildings: geodataframe of building points
        param: claims: geodataframe of NFIP claims
        param: policies: geodataframe of NFIP policies
        """
        # Check that coordinate reference systems agree
        if buildings.crs != self.crs:
            buildings = buildings.to_crs(self.crs)
        if claims.crs != self.crs:
            claims = claims.to_crs(self.crs)
        if policies.crs != self.crs:
            policies = policies.to_crs(self.crs)

        # Convert date columns to pandas datatime format
        claims['Date_of_Loss'] = pd.to_datetime(claims['Date_of_Loss']).dt.tz_localize(None)
        policies['Policy_Effective_Date'] = pd.to_datetime(policies['Policy_Effective_Date']).dt.tz_localize(None)
        policies['Policy_Expiration_Date'] = pd.to_datetime(policies['Policy_Expiration_Date']).dt.tz_localize(None)

        # Get ids of buildings in study area
        buildings_filter = buildings.intersects(self.study_area)
        included_building_ids = buildings[buildings_filter]['building_id'].unique()

        # Get ids of buildings in study area that had a non-zero claim payout during study period
        claims_filter = (claims['building_id'].isin(included_building_ids))
        claims_filter = claims_filter&(claims['Date_of_Loss'] >= self.start_date)
        claims_filter = claims_filter&(claims['Date_of_Loss'] <= self.end_date)
        claims_filter = claims_filter&(claims['Net_Total_Payments'] > 0.0)
        flooded_building_ids = claims[claims_filter]['building_id'].unique()

        # Get ids of buildings in study area that had a policy but no playout during study period
        # (for simplicity, define active policies based on peak date of event)
        policies_filter = (policies['building_id'].isin(included_building_ids))
        policies_filter = policies_filter&(policies['Policy_Effective_Date'] <= self.peak_date)
        policies_filter = policies_filter&(policies['Policy_Expiration_Date'] >= self.peak_date)
        policies_filter = policies_filter&(~policies['building_id'].isin(flooded_building_ids))
        nonflooded_building_ids = policies[policies_filter]['building_id'].unique()

        # Get ids of buildings whose status is unknown (i.e., those who are uninsured)
        building_id_filter = np.isin(included_building_ids,flooded_building_ids,invert=True)
        building_id_filter = building_id_filter&np.isin(included_building_ids,nonflooded_building_ids,invert=True)
        inconclusive_building_ids = included_building_ids[building_id_filter]

        # Create status codes for flooded, not flooded, and inconclusive
        flood_status = np.ones(flooded_building_ids.shape)
        nonflood_status = np.zeros(nonflooded_building_ids.shape)
        inconclusive_status = np.ones(inconclusive_building_ids.shape)*np.nan

        event_building_ids = np.concatenate((flooded_building_ids,nonflooded_building_ids,inconclusive_building_ids))
        event_damage_status = np.concatenate((flood_status,nonflood_status,inconclusive_status))

        # Store properties of flood event
        self.flood_damage_status = pd.DataFrame(data={'building_id':event_building_ids,'flood_damage':event_damage_status})
        self.buildings = buildings[buildings_filter]
        self.claims = claims[claims_filter]
        self.policies = policies[policies_filter]

        # Partition dataset into training (observed) and target (unobserved) datasets
        self.training_dataset = pd.merge(self.buildings,self.flood_damage_status[~self.flood_damage_status['flood_damage'].isna()],on='building_id',how='inner')
        self.target_dataset = pd.merge(self.buildings,self.flood_damage_status[self.flood_damage_status['flood_damage'].isna()],on='building_id',how='inner')

        return(None)

    def preprocess_openfema(self,openfema_claims,openfema_policies):
        """
        param: openfema_claims: geodataframe of openfema claims with census tract geometries
        param: openfema_policies: geodataframe of openfema policies with census tract geometries
        """
        # Check that coordinate reference systems agree
        if openfema_claims.crs != self.crs:
            openfema_claims = openfema_claims.to_crs(self.crs)
        if openfema_policies.crs != self.crs:
            openfema_policies = openfema_policies.to_crs(self.crs)

        # Convert date columns to pandas datatime format
        openfema_claims['dateOfLoss'] = pd.to_datetime(openfema_claims['dateOfLoss']).dt.tz_localize(None)
        openfema_policies['policyEffectiveDate'] = pd.to_datetime(openfema_policies['policyEffectiveDate']).dt.tz_localize(None)
        openfema_policies['policyTerminationDate'] = pd.to_datetime(openfema_policies['policyTerminationDate']).dt.tz_localize(None)

        # Get claims with non-zero payout associated with dates of event
        claims_filter = (openfema_claims['dateOfLoss'] >= self.start_date)
        claims_filter = claims_filter&(openfema_claims['dateOfLoss'] <= self.end_date)
        claims_filter = claims_filter&(openfema_claims['total_payout'] > 0.0)
        openfema_claims = openfema_claims[claims_filter]

        # Get policies in force during time of event
        policies_filter = (openfema_policies['policyEffectiveDate']<=self.peak_date)
        policies_filter = policies_filter&(openfema_policies['policyTerminationDate']>=self.peak_date)
        openfema_policies = openfema_policies[policies_filter]

        # Filter by study region
        openfema_claims = openfema_claims[openfema_claims.intersects(self.study_area)]
        openfema_policies = openfema_policies[openfema_policies.intersects(self.study_area)]

        # Create dummy variable that we'll later use to tally claims
        openfema_claims['claimCount'] = 1

        self.openfema_claims = openfema_claims
        self.openfema_policies = openfema_policies

        return(None)

    def post_stratify(self,stratification_columns):
        """
        Calculate post-stratification weights for training data using OpenFEMA as an auxiliary data source.
        param: stratification_columns: list of columns used to define mutually-exclusive subpopulations (post-strata).
               Note that columns must be present in both OpenFEMA and training datasets.
        """
        training_df = self.training_dataset.copy()
        training_df = training_df[stratification_columns]
        training_df['training_count'] = 1
        training_counts = training_df.groupby(by=stratification_columns).sum().reset_index()

        openfema_columns = stratification_columns.copy()

        if 'flood_damage' in stratification_columns:

            # If choosing to stratify on presence/absence of flood damage, we need information on both
            # claims and policies to determine totals in each post-strata

            openfema_columns.remove('flood_damage')
            claim_counts = self.openfema_claims[openfema_columns + ['claimCount']].groupby(by=openfema_columns).sum().reset_index()
            policy_counts = self.openfema_policies[openfema_columns + ['policyCount']].groupby(by=openfema_columns).sum().reset_index()

            flooded_counts = pd.merge(policy_counts,claim_counts,on=openfema_columns,how='left').fillna(0)
            nonflooded_counts = flooded_counts.copy()

            flooded_counts['flood_damage'] = 1
            nonflooded_counts['flood_damage'] = 0

            flooded_counts['openfema_count'] = flooded_counts['claimCount']
            nonflooded_counts['openfema_count'] = nonflooded_counts['policyCount'] - nonflooded_counts['claimCount']

            openfema_counts = pd.concat([flooded_counts,nonflooded_counts]).drop(columns=['claimCount','policyCount'])

        else:
            # If not choosing to stratify on presence/absence of flood damage,
            # can determine size of post-strata based on policies alone
            openfema_counts = self.openfema_policies[openfema_columns + ['policyCount']].groupby(by=openfema_columns).sum().reset_index()
            openfema_counts = openfema_counts.rename(columns={'policyCount':'openfema_count'})

        openfema_counts = openfema_counts.sort_values(by=stratification_columns).reset_index(drop=True)

        post_strata = pd.merge(openfema_counts,training_counts,on=stratification_columns,how='left').fillna(0)

        post_strata['openfema_fraction'] = post_strata['openfema_count']/post_strata['openfema_count'].sum()
        post_strata['training_fraction'] = post_strata['training_count']/post_strata['training_count'].sum()
        post_strata['weight'] = post_strata['openfema_fraction']/post_strata['training_fraction']

        self.post_strata = post_strata
        self.training_dataset = pd.merge(self.training_dataset,post_strata[stratification_columns + ['weight']],on=stratification_columns,how='left')

        return(None)

    def resample_training_dataset(self,n):
        """
        Use rejection sampling to create a "weighted" version of the training dataset in which the
        frequency of various post-strata reflects that of the openfema dataset.

        For details of method, see 2003 paper by Zadrozny et al. (doi:10.1109/ICDM.2003.1250950)

        param: n: number of samples to include in final "weighted" dataset.
        """

        # g(x) is "proposal" distribution
        g = self.training_dataset.copy()

        # acceptance probability in rejection sampling is proportional to
        # the post-stratification weight assigned to observation
        M = g['weight'].max()
        g['acceptance_prob'] = g['weight']/M

        ng = len(g)
        nf = 0

        df_list = []

        # Perform rejection sampling on g(x) until we reach desired number of
        # observations included in "target" distribution f(x)
        while nf < n:
            u = np.random.uniform(size=ng)
            accept = (u<g['acceptance_prob'])
            df_list.append(g[accept])
            nf += np.sum(accept)

        f = pd.concat(df_list).reset_index(drop=True)

        # At this point we'll likely have slightly more than n observations, so we'll drop some to get n rows
        num_to_remove = nf - n
        drop_indices = np.random.choice(f.index.values, num_to_remove, replace=False)
        f = f.drop(drop_indices).reset_index(drop=True)
        f = f.drop(columns='acceptance_prob')

        self.weighted_training_dataset = f

        return(None)
