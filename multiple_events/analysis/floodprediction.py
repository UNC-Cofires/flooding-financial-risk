import pandas as pd
import numpy as np
import geopandas as gpd
import datetime as dt
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from scipy.interpolate import interp1d

# *** Class for performing Random Forest regression on arbitrary data ***

class RegressionObject:

    def __init__(self,train_df,test_df,target_df,response_variable,features):
        """
        param: train_df: pandas dataframe of training data (m x n+1)
        param: test_df: pandas dataframe of validation data (m x n+1)
        param: target_df: pandas dataframe of target data (z x n)
        param: response_variable: name of response variable
        param: features: list of predictors (n)
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

# Function to impute values of missing attributes in spatial data
def impute_missing_spatially(gdf,columns=None):
    """
    Impute missing values by copying from nearest non-missing neighbor.

    param: gdf: geopandas geodataframe
    param: columns: list of columns to spatially impute missing values for
    """

    # Assess all columns if not specified
    if columns is None:
        columns = gdf.columns

    # Get list of columns with missing values
    nan_columns = []
    for column in columns:
        if gdf[column].isna().sum() > 0:
            nan_columns.append(column)

    # Impute missing values based on nearest non-missing neighbor
    for column in nan_columns:
        m = gdf[column].isna()
        imputed_values = gpd.sjoin_nearest(gdf[m][['geometry']],gdf[~m][[column,'geometry']],how='left')[column]
        gdf.loc[imputed_values.index,column] = imputed_values

    return(gdf)

# Helper function to remove uninformative features
def remove_unnecessary_features(features,data,max_corr=1.0):
    """
    param: features: list of predictors to use in regression
    param: data: pandas dataframe with columns that include features
    param: max_corr: maxiumum absolute correlation allowed between features
    """

    informative_features = []

    # Remove features that have no variation
    for feature in features:
        if len(data[feature].unique()) > 1:
            informative_features.append(feature)

    # Remove features that are highly correlated with another
    corrmat = data[informative_features].corr()

    pair_corr = corrmat.unstack().reset_index()
    pair_corr.columns = ['var1','var2','abs_corr']
    pair_corr['abs_corr'] = np.abs(pair_corr['abs_corr'])
    pair_corr['combo'] = pair_corr.apply(lambda x: '-'.join(np.sort([x['var1'],x['var2']])),axis=1)
    pair_corr = pair_corr.loc[pair_corr['combo'].drop_duplicates().index]
    pair_corr = pair_corr[pair_corr['var1'] != pair_corr['var2']]
    pair_corr = pair_corr.sort_values(by='abs_corr',ascending=False)

    problem_corr = pair_corr[pair_corr['abs_corr'] > max_corr]

    removed_features = []

    while len(problem_corr) > 0:

        var_to_remove = problem_corr.iloc[0]['var2']
        removed_features.append(var_to_remove)

        pair_corr = pair_corr[~((pair_corr['var1']==var_to_remove)|(pair_corr['var2']==var_to_remove))]
        problem_corr = pair_corr[pair_corr['abs_corr'] > max_corr]

    informative_features = [x for x in informative_features if x not in removed_features]

    return(informative_features)

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

    smallnum = np.finfo(float).eps # Small number to prevent float division by zero

    TPR = TP/(P + smallnum)
    TNR = TN/(N + smallnum)
    PPV = TP/(TP + FP + smallnum)

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
    threshold_vals = np.linspace(0,1,101)
    threshold = y_pred[np.argmin(abs_diff(threshold_vals))]
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
    threshold_vals = np.linspace(0,1,101)
    threshold = y_pred[np.argmax(fbeta(threshold_vals))]
    return(threshold)

# *** Flood event class for implementing data processing and prediction workflow

class FloodEvent:

    def __init__(self,study_area,start_date,end_date,peak_date,crs=None,auxiliary_units=None):
        """
        Initialize FloodEvent class.

        param: study_area: geodataframe of areas included in study
        param: start_date: start date of event (YYYY-MM-DD)
        param: end_date: end date of event (YYYY-MM-DD)
        param: peak_date: peak date of event (YYYY-MM-DD)
        param (optional): crs: coordinate reference system to use in analysis
        param (optional): auxiliary_units: geographic units used to tabulate auxiliary data on claim/policy totals
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

        # Define area used to tabulate auxiliary data on claims/policies
        # This can include additional regions that overlap with study area border
        if auxiliary_units is None:
            self.auxiliary_area = self.study_area
        else:
            if auxiliary_units.crs != self.crs:
                auxiliary_units = auxiliary_units.to_crs(self.crs)
            auxiliary_area = auxiliary_units[auxiliary_units.intersects(self.study_area)].dissolve()['geometry'].values[0]
            self.auxiliary_area = auxiliary_area.union(self.study_area)

        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.peak_date = pd.Timestamp(peak_date)

        return(None)

    def preprocess_data(self,parcels,buildings,claims,policies,inflation_multiplier=1.0):
        """
        Determine presence or absence of flooding at building locations based on property-level data

        param: parcels: geodataframe of parcel polygons
        param: buildings: geodataframe of building points
        param: claims: geodataframe of NFIP claims
        param: policies: geodataframe of NFIP policies
        param: inflation_multiplier: multiplier applied to claim damage amounts to account for inflation
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
        buildings_filter = buildings.intersects(self.auxiliary_area)
        studyarea_filter = buildings.intersects(self.study_area)
        buildings['study_area'] = studyarea_filter.astype(int)
        included_building_ids = buildings[buildings_filter]['building_id'].unique()
        studyarea_building_ids = buildings[studyarea_filter]['building_id'].unique()

        # Get ids of parcels in study area
        included_parcel_ids = buildings[buildings_filter]['parcel_id'].unique()
        studyarea_parcel_ids = buildings[studyarea_filter]['parcel_id'].unique()
        parcels_filter = parcels['parcel_id'].isin(included_parcel_ids)
        parcels['study_area'] = parcels['parcel_id'].isin(studyarea_parcel_ids).astype(int)

        # Get ids of buildings in study area that had a non-zero claim payout during study period
        claims['study_area'] = claims['building_id'].isin(studyarea_building_ids).astype(int)
        claims_filter = (claims['building_id'].isin(included_building_ids))
        claims_filter = claims_filter&(claims['Date_of_Loss'] >= self.start_date)
        claims_filter = claims_filter&(claims['Date_of_Loss'] <= self.end_date)
        claims_filter = claims_filter&(claims['Net_Total_Payments'] > 0.0)
        flooded_building_ids = claims[claims_filter]['building_id'].unique()

        # Get ids of buildings in study area that had a policy but no playout during study period
        # (for simplicity, define active policies based on peak date of event)
        policies['study_area'] = policies['building_id'].isin(studyarea_building_ids).astype(int)
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

        # Adjust damages for inflation
        claims['total_payout'] = claims['total_payout']*inflation_multiplier

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

    def preprocess_auxiliary(self,auxiliary_claims,auxiliary_policies,unit_name='censusTract',inflation_multiplier=1.0):
        """
        param: auxiliary_claims: pandas dataframe of auxiliary claims data from OpenFEMA
        param: auxiliary_policies: pandas dataframe of auxiliary claims data from OpenFEMA / EDF
        param: unit_name: column name corresponding to geographic unit to which auxiliary data can be located
        param: inflation_multiplier: multiplier applied to claim damage amounts to account for inflation
        """

        # Convert date columns to pandas datatime format
        auxiliary_claims['dateOfLoss'] = pd.to_datetime(auxiliary_claims['dateOfLoss']).dt.tz_localize(None)
        auxiliary_policies['policyEffectiveDate'] = pd.to_datetime(auxiliary_policies['policyEffectiveDate']).dt.tz_localize(None)
        auxiliary_policies['policyTerminationDate'] = pd.to_datetime(auxiliary_policies['policyTerminationDate']).dt.tz_localize(None)

        # Get claims with non-zero payout associated with dates of event
        claims_filter = (auxiliary_claims['dateOfLoss'] >= self.start_date)
        claims_filter = claims_filter&(auxiliary_claims['dateOfLoss'] <= self.end_date)
        claims_filter = claims_filter&(auxiliary_claims['total_payout'] > 0.0)
        auxiliary_claims = auxiliary_claims[claims_filter]

        # Get policies in force during time of event
        policies_filter = (auxiliary_policies['policyEffectiveDate']<=self.peak_date)
        policies_filter = policies_filter&(auxiliary_policies['policyTerminationDate']>=self.peak_date)
        auxiliary_policies = auxiliary_policies[policies_filter]

        # Filter by study region
        included_units = self.buildings[unit_name].unique()
        auxiliary_claims = auxiliary_claims[auxiliary_claims[unit_name].isin(included_units)]
        auxiliary_policies = auxiliary_policies[auxiliary_policies[unit_name].isin(included_units)]

        # Create dummy variable that we'll later use to tally claims
        # (policy data already includes a policyCount variable)
        auxiliary_claims['claimCount'] = 1

        # Adjust damages for inflation
        auxiliary_claims['total_payout'] = auxiliary_claims['total_payout']*inflation_multiplier

        self.auxiliary_claims = auxiliary_claims
        self.auxiliary_policies = auxiliary_policies

        return(None)

    def stratify_missing(self,stratification_columns):
        """
        Determine number of missing flooded and non-flooded buildings within each user-defined strata by
        comparing against claim and policy counts from auxiliary data sources (e.g., OpenFEMA, EDF).

        param: stratification_columns: list of columns used to define mutually-exclusive subpopulations (strata).
               Note that columns must be present in both auxiliary and training datasets.
        """

        # Calculate number of flooded/nonflooded buildings within each strata in address-level dataset
        training_df = self.training_dataset.copy()
        training_df = training_df[stratification_columns + ['flood_damage']].rename(columns={'flood_damage':'training_flooded'})
        training_df['training_nonflooded'] = 1 - training_df['training_flooded']
        training_counts = training_df.groupby(by=stratification_columns).sum().reset_index()

        # Calculate number of flooded/nonflooded buildings within each strata in OpenFEMA dataset
        claim_counts = self.auxiliary_claims[stratification_columns + ['claimCount']].groupby(by=stratification_columns).sum().reset_index()
        policy_counts = self.auxiliary_policies[stratification_columns + ['policyCount']].groupby(by=stratification_columns).sum().reset_index()

        auxiliary_counts = pd.merge(policy_counts,claim_counts,on=stratification_columns,how='left').fillna(0)
        auxiliary_counts['auxiliary_flooded'] = auxiliary_counts['claimCount'].astype(int)
        auxiliary_counts['auxiliary_nonflooded'] = auxiliary_counts['policyCount'] - auxiliary_counts['claimCount']
        auxiliary_counts['auxiliary_nonflooded'] = auxiliary_counts['auxiliary_nonflooded'].apply(lambda x: max(x,0)).astype(int)
        auxiliary_counts = auxiliary_counts.drop(columns=['claimCount','policyCount'])

        strata_counts = pd.merge(auxiliary_counts,training_counts,on=stratification_columns,how='left').fillna(0)
        strata_counts['missing_flooded'] = strata_counts['auxiliary_flooded'] - strata_counts['training_flooded']
        strata_counts['missing_nonflooded'] = strata_counts['auxiliary_nonflooded'] - strata_counts['training_nonflooded']
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

        existing_data = self.training_dataset.copy()
        existing_data['pseudo_absence'] = False

        # Buildings that we could select as pseudo absences
        potential_pseudo_absences = self.target_dataset[self.target_dataset['strata'].isin(self.strata_counts.index)]

        for i in range(n_realizations):
            pseudo_absences = potential_pseudo_absences.groupby('strata',group_keys=False).apply(sampling_func)
            pseudo_absences['pseudo_absence'] = True
            pseudo_absences['flood_damage'] = 0
            df_list.append(existing_data)
            df_list.append(pseudo_absences)

        self.adjusted_training_dataset = pd.concat(df_list).reset_index(drop=True)

        return(None)

    def crop_to_study_area(self):
        """
        Drop records in training / target datasets that fall outside of study area.

        Because the study area boundaries don't necessarily line up exactly with the geographic units
        used to tabulate auxiliary data on claims / policy counts, we initially need to include some additional
        regions that fall outside of study area when creating pseudo-absences.

        Before we actually train our model, we want to remove any records outside study area.
        """

        self.training_dataset = self.training_dataset[self.training_dataset['study_area']==1].reset_index(drop=True)
        self.adjusted_training_dataset = self.adjusted_training_dataset[self.adjusted_training_dataset['study_area']==1].reset_index(drop=True)
        self.target_dataset = self.target_dataset[self.target_dataset['study_area']==1].reset_index(drop=True)
        return(None)

    def cross_validate(self,response_variable,features,k=5,use_adjusted=True,threshold=None):
        """
        param: response_variable: name of response variable
        param: features: list of predictors
        param: k: number of k-fold cross validation iterations to perform
        param: use_adjusted: if true, train models using adjusted training data which includes pseudo-absences
                            (note that we still test on genuine observations)
        param: threshold: probability threshold used for classification
        """

        features = [f for f in features if f != response_variable]
        kf = KFold(n_splits=k,random_state=None,shuffle=True)

        if threshold is None:
            compute_threshold = True
        else:
            compute_threshold = False

        results_list = []

        fpr_viz_vals = np.linspace(0,1,501)
        tpr_viz_vals = np.zeros((k,fpr_viz_vals.shape[0]))
        roc_labels = []

        rec_viz_vals = np.linspace(0,1,501)
        prec_viz_vals = np.zeros((k,rec_viz_vals.shape[0]))
        pr_labels = []

        for i,(train_indices,test_indices) in enumerate(kf.split(self.training_dataset)):

            results_dict = {'fold':i}

            test_df = self.training_dataset.iloc[test_indices].copy()

            if use_adjusted:

                # Exclude buildings included in test set from training data
                test_building_ids = test_df['building_id']
                m1 = self.adjusted_training_dataset['building_id'].isin(test_building_ids)

                # Randomly drop 1/k of pseudo-absences to maintain balance
                m2 = self.adjusted_training_dataset['pseudo_absence']
                m3 = np.random.rand(len(self.adjusted_training_dataset)) < 1/k

                records_to_drop = m1|(m2&m3)
                train_df = self.adjusted_training_dataset[~records_to_drop].copy()

            else:
                train_df = self.training_dataset.iloc[train_indices].copy()

            # Fit model
            mod = RegressionObject(train_df,test_df,test_df,response_variable,features)
            mod.model_fit()

            # If not already specified, calculate the optimal threshold based on
            # training set (note that this includes pseudo-absences)
            if compute_threshold:
                y_pred_threshold = mod.model_predict(mod.x)
                y_true_threshold = mod.y
                threshold = minimized_difference_threshold(y_pred_threshold,y_true_threshold)

            results_dict['threshold'] = threshold

            # Predict class probabilities for test set
            y_pred = mod.model_predict(mod.x_test)

            # Get actual class labels for test set
            y_true = mod.y_test

            # Assign class labels to predictions
            y_class = mod.model_classify(mod.x_test,threshold)

            # Compute performance metrics

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

            # Get data needed for ROC and PR curves
            fpr_vals, tpr_vals, threshold_vals = metrics.roc_curve(y_true, y_pred)
            roc_interp_func = interp1d(fpr_vals,tpr_vals)
            tpr_viz_vals[i,:] = roc_interp_func(fpr_viz_vals)
            auc_rounded = np.round(results_dict['roc_auc'],2)
            roc_labels.append(f'CV fold {i+1} (AUC={auc_rounded})')

            prec_vals, rec_vals, threshold_vals = metrics.precision_recall_curve(y_true, y_pred)
            pr_interp_func = interp1d(rec_vals,prec_vals)
            prec_viz_vals[i,:] = pr_interp_func(rec_viz_vals)
            ap_rounded = np.round(results_dict['avg_prec'],2)
            pr_labels.append(f'CV fold {i+1} (AP={ap_rounded})')

            results_list.append(results_dict.copy())

        # Save performance metrics in dataframe
        results_df = pd.DataFrame(results_list)

        # Create ROC curve plot
        roc_mean = np.round(results_df['roc_auc'].mean(),2)
        roc_std = np.round(results_df['roc_auc'].std(),2)

        tpr_mean = tpr_viz_vals.mean(axis=0)
        tpr_std = tpr_viz_vals.std(axis=0)

        roc_fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,6))
        ax.set_aspect('equal', adjustable='box')

        for i in range(k):
            ax.plot(fpr_viz_vals,tpr_viz_vals[i],alpha=0.65,label=roc_labels[i])

        ax.plot(fpr_viz_vals,tpr_mean,'k',lw=2,label=f'Mean ROC (AUC={roc_mean}±{roc_std})')
        ax.fill_between(fpr_viz_vals, tpr_mean - tpr_std, tpr_mean + tpr_std,color='gray',alpha=0.2,label=f'±1 std. dev.')
        ax.plot([0,1],[0,1],'k--',alpha=0.65,label='Random classifier (AUC=0.5)')

        ax.set_xlim([0,1])
        ax.set_ylim([0,1])

        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title('Receiver operating characteristic (ROC) curve')

        ax.legend()
        roc_fig.tight_layout()

        # Create PR curve plot

        pr_mean = np.round(results_df['avg_prec'].mean(),2)
        pr_std = np.round(results_df['avg_prec'].std(),2)

        prec_mean = prec_viz_vals.mean(axis=0)
        prec_std = prec_viz_vals.std(axis=0)

        pr_random = np.round(self.training_dataset[response_variable].mean(),2)

        pr_fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,6))

        for i in range(k):
            ax.plot(rec_viz_vals,prec_viz_vals[i],alpha=0.65,label=pr_labels[i])

        ax.plot(rec_viz_vals,prec_mean,'k',lw=2,label=f'Mean PR (AP={pr_mean}±{pr_std})')
        ax.fill_between(rec_viz_vals, prec_mean - prec_std, prec_mean + prec_std,color='gray',alpha=0.2,label=f'±1 std. dev.')
        ax.plot([0,1],[pr_random,pr_random],'k--',alpha=0.65,label=f'Random classifier (AP={pr_random})')

        ax.set_xlim([0,1])
        ax.set_ylim([None,1])

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall (PR) curve')

        ax.legend()
        pr_fig.tight_layout()

        self.performance_metrics = results_df
        self.roc_curve = roc_fig
        self.pr_curve = pr_fig

        return(None)

    def predict_presence_absence(self,response_variable,features,use_adjusted=True,threshold=0.5):
        """
        Predict the presence/absence of flooding at target locations.

        param: response_variable: name of response variable
        param: features: list of predictors
        param: use_adjusted: if true, train models using adjusted training data which includes pseudo-absences
                            (note that we still test on genuine observations)
        param: threshold: probability threshold used for classification
        """

        if use_adjusted:
            mod = RegressionObject(self.adjusted_training_dataset,self.adjusted_training_dataset,self.target_dataset,response_variable,features)
        else:
            mod = RegressionObject(self.training_dataset,self.training_dataset,self.target_dataset,response_variable,features)

        # Fit Random Forest model
        mod.model_fit()

        # Predict presence of flood damage
        self.target_dataset[f'{response_variable}_prob'] = mod.model_predict(mod.x_target)
        self.target_dataset[f'{response_variable}_class'] = mod.model_classify(mod.x_target,threshold)

        # Get feature importance
        importances = mod.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in mod.model.estimators_], axis=0)
        importance_order = np.argsort(importances)

        importances = importances[importance_order]
        std = std[importance_order]
        sorted_features = np.array(features)[importance_order]

        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,4))
        ax.barh(sorted_features, importances, xerr=std, align='center',color='C0',alpha=0.5,lw=1,edgecolor='k')
        ax.set_xlim([0,None])
        ax.set_xlabel('Mean decrease in impurity (MDI)')
        ax.set_title('Random Forest Feature Importance')

        self.presence_absence_features = features
        self.presence_absence_response_variable = response_variable
        self.presence_absence_feature_importance = fig
        self.presence_absence_model = mod

        return(None)

    def predict_damage_cost(self,presence_variable,cost_variable,features):
        """
        Predict the cost of flood damage among flooded homes.

        param: presence_variable: name of variable denoting presence of flood damage
        param: cost_variable: name of variable denoting cost of flood damage
        param: features: list of predictors
        """

        # If more than one claim matches to a building, take one with higher payout
        payouts_by_building = self.claims[['building_id',cost_variable]].groupby('building_id').max().reset_index()

        self.training_dataset[cost_variable] = self.training_dataset[['building_id']].merge(payouts_by_building,how='left',on='building_id')[cost_variable].fillna(0)

        damage_cost_training = self.training_dataset[(self.training_dataset[presence_variable]==1)]
        damage_cost_target = self.target_dataset[(self.target_dataset[f'{presence_variable}_class']==1)]

        mod = RegressionObject(damage_cost_training,damage_cost_training,damage_cost_target,cost_variable,features)

        # Fit Random Forest model
        mod.model_fit()

        # Compute R^2 using training data
        y_pred = mod.model_predict(mod.x_test)
        y_true = mod.y_test
        errors = y_true - y_pred
        SSE = np.sum(errors**2)
        SST = np.sum((y_true - np.mean(y_true))**2)
        R_sq = 1 - SSE/SST
        RMSE = np.sqrt(np.mean(errors**2))

        self.damage_cost_RMSE = RMSE
        self.damage_cost_R_sq = R_sq

        # Predict cost of damage among buildings predicted as flooded
        damage_cost_target[cost_variable] = mod.model_predict(mod.x_target)

        # Join back to target dataset
        self.target_dataset[cost_variable] = self.target_dataset[['building_id']].merge(damage_cost_target[['building_id',cost_variable]],how='left',on='building_id')[cost_variable].fillna(0)

        # Save info
        self.damage_cost_features = features
        self.damage_cost_response_variable = cost_variable
        self.damage_cost_model = mod

        # Get feature importance
        importances = mod.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in mod.model.estimators_], axis=0)
        importance_order = np.argsort(importances)

        importances = importances[importance_order]
        std = std[importance_order]
        sorted_features = np.array(features)[importance_order]

        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,4))
        ax.barh(sorted_features, importances, xerr=std, align='center',color='C0',alpha=0.5,lw=1,edgecolor='k')
        ax.set_xlim([0,None])
        ax.set_xlabel('Mean decrease in impurity (MDI)')
        ax.set_title('Random Forest Feature Importance')

        self.damage_cost_feature_importance = fig

        return(None)

    def aggregate_flood_damage(self,stratification_columns):
        """
        Combine estimates of insured (i.e., training) and uninsured (i.e., target) losses to create
        overall estimate of flood damage in study region.

        param: stratification_columns: list of columns used to aggregate damage estimates.
        """
        insured_df = self.training_dataset[['building_id','study_area','geometry','flood_damage','total_payout'] + stratification_columns].rename(columns={'flood_damage':'flood_damage_class'})
        insured_df = insured_df[insured_df['study_area']==1]
        insured_df['flood_damage_prob'] = insured_df['flood_damage_class']
        insured_df['insured'] = 1
        uninsured_df = self.target_dataset[['building_id','study_area','geometry','flood_damage_prob','flood_damage_class','total_payout'] + stratification_columns]
        uninsured_df = uninsured_df[uninsured_df['study_area']==1]
        uninsured_df['insured'] = 0
        combined_df = pd.concat([insured_df,uninsured_df]).reset_index(drop=True)
        combined_columns = ['building_id','study_area'] + stratification_columns + ['insured','flood_damage_prob','flood_damage_class','total_payout','geometry']
        combined_df = combined_df[combined_columns]

        agg_dict = {'building_id':'count','flood_damage_class':'sum','total_payout':'sum'}
        insured_rename_dict = {'building_id':'n_insured','flood_damage_class':'n_flooded_insured','total_payout':'cost_flooded_insured'}
        uninsured_rename_dict = {'building_id':'n_uninsured','flood_damage_class':'n_flooded_uninsured','total_payout':'cost_flooded_uninsured'}
        combined_rename_dict = {'building_id':'n_total','flood_damage_class':'n_flooded_total','total_payout':'cost_flooded_total'}

        insured_agg = insured_df[stratification_columns + list(agg_dict.keys())].groupby(stratification_columns).agg(agg_dict).rename(columns=insured_rename_dict)
        uninsured_agg = uninsured_df[stratification_columns + list(agg_dict.keys())].groupby(stratification_columns).agg(agg_dict).rename(columns=uninsured_rename_dict)
        combined_agg = combined_df[stratification_columns + list(agg_dict.keys())].groupby(stratification_columns).agg(agg_dict).rename(columns=combined_rename_dict)

        agg_df = insured_agg.join(uninsured_agg,how='outer').join(combined_agg,how='outer')
        agg_df = agg_df[['n_insured','n_uninsured','n_total','n_flooded_insured','n_flooded_uninsured','n_flooded_total','cost_flooded_insured','cost_flooded_uninsured','cost_flooded_total']].reset_index()
        agg_df = agg_df.sort_values(by=stratification_columns)

        return(agg_df,combined_df)
