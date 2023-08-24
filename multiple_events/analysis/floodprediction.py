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
    Calculate the optimal threshold (cut-point) for classification based on
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

def maximized_fbeta_threshold(y_pred,y_true,beta=1):
    """
    Calculate the optimal threshold (cut-point) for classification that
    maximized the f_beta score (beta=1 equivalent to f1 score).

    param: y_pred: numpy array of predicted class probabilities
    param: y_true: numpy array of true class labels
    param: beta: relative importance of recall vs precision
    """
    fbeta = np.vectorize(lambda x: metrics.fbeta_score(y_true, (y_pred > x).astype(int), beta=beta))
    threshold = y_pred[np.argmax(fbeta(y_pred))]
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

    def cross_validate(self,threshold=None):
        """
        Perform k-fold cross validation

        param: threshold: probability threshold (cut-point) for classification
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

            if threshold is None:
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

    def preprocess_data(self,parcels,buildings,claims,policies):
        """
        Determine presence or absence of flooding at building locations based on property-level data

        param: parcels: geodataframe of parcel polygons
        param: buildings: geodataframe of building points
        param: claims: geodataframe of NFIP claims
        param: policies: geodataframe of NFIP policies
        """
        # Check that coordinate reference systems agree
        if parcels.crs != self.crs:
            parcels = parcels.to_crs(self.crs)
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

        # Get ids of parcels in study area
        included_parcel_ids = buildings[buildings_filter]['parcel_id'].unique()
        parcels_filter = parcels['parcel_id'].isin(included_parcel_ids)

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
        self.parcels = parcels[parcels_filter]
        self.buildings = buildings[buildings_filter]
        self.claims = claims[claims_filter]
        self.policies = policies[policies_filter]

        # Partition dataset into training (observed) and target (unobserved) datasets
        self.training_dataset = pd.merge(self.buildings,self.flood_damage_status[~self.flood_damage_status['flood_damage'].isna()],on='building_id',how='inner')
        self.target_dataset = pd.merge(self.buildings,self.flood_damage_status[self.flood_damage_status['flood_damage'].isna()],on='building_id',how='inner')

        return(None)

    def preprocess_openfema(self,openfema_claims,openfema_policies):
        """
        param: openfema_claims: pandas dataframe of openfema claims
        param: openfema_policies: pandas dataframe of openfema policies
        """

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
        included_tracts = self.buildings['censusTract'].unique()
        openfema_claims = openfema_claims[openfema_claims['censusTract'].isin(included_tracts)]
        openfema_policies = openfema_policies[openfema_policies['censusTract'].isin(included_tracts)]

        # Create dummy variable that we'll later use to tally claims
        # (OpenFEMA policy data already includes a policyCount variable)
        openfema_claims['claimCount'] = 1

        self.openfema_claims = openfema_claims
        self.openfema_policies = openfema_policies

        return(None)

    def stratify_missing(self,stratification_columns):
        """
        Determine number of missing flooded and non-flooded buildings within each user-defined strata by
        comparing against claim/policy counts from OpenFEMA.

        param: stratification_columns: list of columns used to define mutually-exclusive subpopulations (strata).
               Note that columns must be present in both OpenFEMA and training datasets.
        """

        # Calculate number of flooded/nonflooded buildings within each strata in address-level dataset
        training_df = self.training_dataset.copy()
        training_df = training_df[stratification_columns + ['flood_damage']].rename(columns={'flood_damage':'training_flooded'})
        training_df['training_nonflooded'] = 1 - training_df['training_flooded']
        training_counts = training_df.groupby(by=stratification_columns).sum().reset_index()

        # Calculate number of flooded/nonflooded buildings within each strata in OpenFEMA dataset
        claim_counts = self.openfema_claims[stratification_columns + ['claimCount']].groupby(by=stratification_columns).sum().reset_index()
        policy_counts = self.openfema_policies[stratification_columns + ['policyCount']].groupby(by=stratification_columns).sum().reset_index()

        openfema_counts = pd.merge(policy_counts,claim_counts,on=stratification_columns,how='left').fillna(0)
        openfema_counts['openfema_flooded'] = openfema_counts['claimCount'].astype(int)
        openfema_counts['openfema_nonflooded'] = openfema_counts['policyCount'] - openfema_counts['claimCount']
        openfema_counts['openfema_nonflooded'] = openfema_counts['openfema_nonflooded'].apply(lambda x: max(x,0)).astype(int)
        openfema_counts = openfema_counts.drop(columns=['claimCount','policyCount'])

        strata_counts = pd.merge(openfema_counts,training_counts,on=stratification_columns,how='left').fillna(0)
        strata_counts['missing_flooded'] = strata_counts['openfema_flooded'] - strata_counts['training_flooded']
        strata_counts['missing_nonflooded'] = strata_counts['openfema_nonflooded'] - strata_counts['training_nonflooded']
        strata_counts['missing_flooded'] = strata_counts['missing_flooded'].apply(lambda x: max(x,0)).astype(int)
        strata_counts['missing_nonflooded'] = strata_counts['missing_nonflooded'].apply(lambda x: max(x,0)).astype(int)

        strata_counts['strata'] = strata_counts[stratification_columns].astype(str).apply(lambda x: '_'.join(x),axis=1)
        self.training_dataset['strata'] = self.training_dataset[stratification_columns].astype(str).apply(lambda x: '_'.join(x),axis=1)
        self.target_dataset['strata'] = self.target_dataset[stratification_columns].astype(str).apply(lambda x: '_'.join(x),axis=1)
        self.strata_counts = strata_counts.set_index('strata')

        return(None)

    def create_pseudo_absences(self,n_realizations=1):
        """
        Create synthetic nulls (i.e., pseudo-absences) so that within-strata counts of
        non-flooded buildings match those in OpenFEMA dataset.

        param: n_realizations: number of times to randomly sample pseudo absences from available buildings
                               (this will increase the size of the adjusted training dataset)
        """
        sampling_func = lambda x: x.sample(min(self.strata_counts.loc[x.name,'missing_nonflooded'],len(x)))

        df_list = []

        for i in range(n_realizations):
            existing_data = self.training_dataset.copy()
            pseudo_absences = self.target_dataset.groupby('strata',group_keys=False).apply(sampling_func)
            existing_data['pseudo_absence'] = False
            pseudo_absences['pseudo_absence'] = True
            df_list.append(existing_data)
            df_list.append(pseudo_absences)

        self.adjusted_training_dataset = pd.concat(df_list).reset_index(drop=True)

        return(None)
