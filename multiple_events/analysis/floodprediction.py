import pandas as pd
import numpy as np
import geopandas as gpd
import datetime as dt
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import KFold,GridSearchCV
from scipy.interpolate import interp1d
import time

# *** Class for performing Random Forest regression on arbitrary data ***

class RegressionObject:

    def __init__(self,train_df,test_df,target_df,response_variable,features,n_cores=1,hyperparams={'max_depth':5}):
        """
        param: train_df: pandas dataframe of training data (m x n+1)
        param: test_df: pandas dataframe of validation data (m x n+1)
        param: target_df: pandas dataframe of target data (z x n)
        param: response_variable: name of response variable
        param: features: list of predictors (n)
        param: n_cores: number of threads to use for parallelization of model
        param: hyperparams: hyperparameters passed to random forest model
        """
        self.features = [f for f in features if f != response_variable and f != train_df.index.name]
        self.response_variable = response_variable
        self.x = train_df[self.features].to_numpy()
        self.y = train_df[self.response_variable].to_numpy()
        self.x_test = test_df[self.features].to_numpy()
        self.y_test = test_df[self.response_variable].to_numpy()
        self.x_target = target_df[self.features].to_numpy()
        self.model = None
        self.n_cores = n_cores
        self.hyperparams = hyperparams
        return(None)

    def model_fit(self):
        """
        Fit a random forest regression model to the data
        """
        self.model = RandomForestRegressor(**self.hyperparams,n_jobs=self.n_cores)
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

    TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_class,labels=[0,1]).ravel()

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
    threshold = threshold_vals[np.argmin(abs_diff(threshold_vals))]
    return(threshold)

def maximized_accuracy_threshold(y_pred,y_true):
    """
    Calculate the optimal threshold (cut-point) for classification that
    maximized accuracy.

    param: y_pred: numpy array of predicted class probabilities
    param: y_true: numpy array of true class labels
    """
    accuracy = np.vectorize(lambda x: metrics.accuracy_score(y_true, (y_pred > x).astype(int)))
    threshold_vals = np.linspace(0,1,101)
    threshold = threshold_vals[np.argmax(accuracy(threshold_vals))]
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
    threshold = threshold_vals[np.argmax(fbeta(threshold_vals))]
    return(threshold)

# Helper function to format elapsed time in seconds
def format_elapsed_time(seconds):
    seconds = int(np.round(seconds))
    hours = seconds // 3600
    seconds = seconds - hours*3600
    minutes = seconds // 60
    seconds = seconds - minutes*60
    return(f'{hours}h:{minutes:02d}m:{seconds:02d}s')

# Cross validation setup
def cv_fold(fold,train_df,test_df,presence_response_variable,presence_features,cost_response_variable,cost_features,n_cores=1):
        """
        param: fold: fold number
        param: train_df: training data
        param: test_df: testing data
        param: presence_response_variable: name of binary response variable indicating presence/absence of flooding
        param: presence_features: list of features used to predict the presence/absence of flood damage
        param: cost_response_variable: name of continuous variable indicating cost of damages
        param: cost_features: list of features used to predict the cost of damage to flooded structures
        param: n_cores: number of cores to use if running tasks in parallel
        """
        
        test_df['fold'] = fold
        
        fpr_viz_vals = np.linspace(0,1,501)
        rec_viz_vals = np.linspace(0,1,501)
        
        # Determine hyperparameters for random forest models
        presence_hyperparams,cost_hyperparams = tune_hyperparams(train_df,presence_response_variable,presence_features,cost_response_variable,cost_features,n_cores=n_cores)
        
        # Fit prediction model for presence/absence of flooding
        presence_mod = RegressionObject(train_df,test_df,test_df,presence_response_variable,presence_features,n_cores=n_cores,hyperparams=presence_hyperparams)
        presence_mod.model_fit()
        
        # If not already specified, calculate the optimal threshold based on
        # the training set (note that this includes pseudo-absences)
        y_pred_threshold = presence_mod.model_predict(presence_mod.x)
        y_true_threshold = presence_mod.y
        threshold = maximized_accuracy_threshold(y_pred_threshold,y_true_threshold)        
        
        # Predict presence/absence for test set
        y_pred = presence_mod.model_predict(presence_mod.x_test)

        # Get actual class labels for test set
        y_true = presence_mod.y_test

        # Assign class labels to predictions
        y_class = presence_mod.model_classify(presence_mod.x_test,threshold)
        
        # Add predictions to dataset
        test_df[f'{presence_response_variable}_prob'] = y_pred
        test_df[f'{presence_response_variable}_class'] = y_class
        
        # Fit prediction model for cost of damage among flooded homes
        cost_train_df = train_df[train_df[presence_response_variable]==1]
        cost_mod = RegressionObject(cost_train_df,test_df,test_df,cost_response_variable,cost_features,n_cores=n_cores,hyperparams=cost_hyperparams)
        cost_mod.model_fit()
        
        # Predict cost of flood damage among buildings in test set
        c_pred = cost_mod.model_predict(cost_mod.x_test)
        
        # Assume buildings classified as non-flooded don't incur damage costs
        c_pred = y_class*c_pred 
        
        # Get actual cost of flood damage for test set
        c_true = cost_mod.y_test
        
        # Add predictions to dataset
        test_df[f'{cost_response_variable}_pred'] = c_pred
            
        return(test_df)
    
# Measures of model performance
def performance_metrics(y_pred,y_class,y_true,c_pred,c_true):
    """
    param: y_pred: numpy array of predicted flood damage probabilities
    param: y_class: numpy array of predicted flood damage class labels
    param: y_true: numpy array of true flood damage class labels
    param: c_pred: numpy array of predicted damage costs
    param: c_true: numpy array of true damage costs
    """
    results_dict = {}
    fpr_viz_vals = np.linspace(0,1,501)
    rec_viz_vals = np.linspace(0,1,501)
    
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

    # Metrics related to damage cost estimation
    R_sq = metrics.r2_score(c_true,c_pred)
    MAE = metrics.mean_absolute_error(c_true,c_pred)
    RMSE = metrics.mean_squared_error(c_true,c_pred,squared=False)
    results_dict['damage_cost_Rsq'] = R_sq
    results_dict['damage_cost_MAE'] = MAE
    results_dict['damage_cost_RMSE'] = RMSE
    
    # Get data needed for ROC and PR curves
    fpr_vals, tpr_vals, threshold_vals = metrics.roc_curve(y_true, y_pred)
    roc_interp_func = interp1d(fpr_vals,tpr_vals)
    tpr_viz_vals = roc_interp_func(fpr_viz_vals)
    roc_df = pd.DataFrame({'fpr':fpr_viz_vals,'tpr':tpr_viz_vals})

    prec_vals, rec_vals, threshold_vals = metrics.precision_recall_curve(y_true, y_pred)
    pr_interp_func = interp1d(rec_vals,prec_vals)
    prec_viz_vals = pr_interp_func(rec_viz_vals)
    pr_df = pd.DataFrame({'rec':rec_viz_vals,'prec':prec_viz_vals})
    
    return(results_dict,roc_df,pr_df)

def tune_hyperparams(data,presence_response_variable,presence_features,cost_response_variable,cost_features,k=5,n_cores=1):
    """
    param: data: training data used for hyperparameter tuning
    param: presence_response_variable: name of binary response variable indicating presence/absence of flooding
    param: presence_features: list of features used to predict the presence/absence of flood damage
    param: cost_response_variable: name of continuous variable indicating cost of damages
    param: cost_features: list of features used to predict the cost of damage to flooded structures
    param: use_adjusted: if true, train models using adjusted training data which includes pseudo-absences
    param: k: number of folds to use in hyperparameter tuning
    param: n_cores: number of cores to use if running tasks in parallel
    """        
        
    # Determine optimal hyperparameters for presence-absence model
    y = data[presence_response_variable].to_numpy()
    x = data[presence_features].to_numpy()

    param_grid = {'n_estimators':[200],'max_features':[1.0],'max_depth':[5,7,9]}
    model = RandomForestRegressor()

    grid_search = GridSearchCV(model, param_grid=param_grid, cv=k, scoring='neg_mean_squared_error',n_jobs=n_cores)
    grid_search.fit(x,y)
    presence_hyperparams = grid_search.best_params_ 

    # Determine optimal hyperparameters for damage cost model
    m = (data[presence_response_variable]==1)
    y = data[m][cost_response_variable].to_numpy()
    x = data[m][cost_features].to_numpy()

    param_grid = {'n_estimators':[200],'max_features':[1.0],'max_depth':[5,7,9]}
    model = RandomForestRegressor()

    grid_search = GridSearchCV(model, param_grid=param_grid, cv=k, scoring='neg_mean_squared_error',n_jobs=n_cores)
    grid_search.fit(x,y)
    cost_hyperparams = grid_search.best_params_

    return(presence_hyperparams,cost_hyperparams)
    
def build_neighbor_dict(gdf):
    """
    Build a dictionary specifying the neighbors of a given polygon in a geodataframe
    """
    neighbor_dict = {}
    
    gdf['geometry'] = gdf['geometry'].buffer(5)
    
    for index, row in gdf.iterrows():
        neighbors = gdf[gdf['geometry'].intersects(row['geometry'])].index.tolist()
        neighbors = [x for x in neighbors if x != index]
        neighbor_dict[index] = neighbors
        
    return(neighbor_dict)

def draw_k_value_1_degrees(tile_index,tile_ks,neighbor_dict,k_vals):
    """
    Draw a value of k for a given tile, ensuring that identical values of k have ≥1 degree of separation
    Must have at least 4 unique values of k to select from.  
    """
    neighbor_ks = tile_ks[neighbor_dict[tile_index]]
    k_options = k_vals[~np.isin(k_vals,neighbor_ks)]
    return(np.random.choice(k_options))

def draw_k_value_2_degrees(tile_index,tile_ks,neighbor_dict,k_vals):
    """
    Draw a value of k for a given tile, ensuring that identical values of k have ≥2 degrees of separation
    Must have at least 10 unique values of k to select from.  
    """
    neighbor_ks = tile_ks[neighbor_dict[tile_index]]
    
    for neighbor_index in neighbor_dict[tile_index]:
        neighbor_ks = np.concatenate((neighbor_ks,tile_ks[neighbor_dict[neighbor_index]]))
        
    neighbor_ks = np.unique(neighbor_ks)
    
    k_options = k_vals[~np.isin(k_vals,neighbor_ks)]
    return(np.random.choice(k_options))

def draw_k_value_3_degrees(tile_index,tile_ks,neighbor_dict,k_vals):
    """
    Draw a value of k for a given tile, ensuring that identical values of k have ≥3 degrees of separation
    Must have at least 19 unique values of k to select from.  
    """
    neighbor_ks = tile_ks[neighbor_dict[tile_index]]
    
    for i in neighbor_dict[tile_index]:
        neighbor_ks = np.concatenate((neighbor_ks,tile_ks[neighbor_dict[i]]))
        for j in neighbor_dict[i]:
            neighbor_ks = np.concatenate((neighbor_ks,tile_ks[neighbor_dict[j]]))
        
    neighbor_ks = np.unique(neighbor_ks)
    
    k_options = k_vals[~np.isin(k_vals,neighbor_ks)]
    return(np.random.choice(k_options))

# Rather than all this, maybe just do leave location out since it seems like each evaluation is really fast
def spatial_kfold(tiles,k=100,degree=3):
    """
    param: tiles: pandas geodataframe of spatial blocks (usually hexagonal tiles)
    param: k: number of cross-validation folds
    param: degree: number of degrees of separation between blocks in same fold (can be 1, 2, or 3)
    """
    
    if degree not in [1,2,3]:
        raise(ValueError('Parameter `degree` must have value of 1, 2, or 3'))
        
    if degree==3:
        draw_k_value = lambda w,x,y,z: draw_k_value_3_degrees(w,x,y,z)
        if k < 19:
            raise(ValueError('Parameter k must be ≥19 when degree=3'))
    if degree==2:
        draw_k_value = lambda w,x,y,z: draw_k_value_2_degrees(w,x,y,z)
        if k < 10:
            raise(ValueError('Parameter `k` must be ≥10 when degree=2'))
    else:
        draw_k_value = lambda w,x,y,z: draw_k_value_1_degrees(w,x,y,z)
        if k < 4:
            raise(ValueError('Parameter `k` must be ≥4 when degree=1'))
            
    num_tiles = len(tiles)
    tiles.index = np.arange(num_tiles)
    k_vals = np.arange(k)
    
    tile_ks = -1*np.ones(num_tiles,dtype=int)

    neighbor_dict = build_neighbor_dict(tiles)

    for i in range(num_tiles):
        tile_ks[i] = draw_k_value(i,tile_ks,neighbor_dict,k_vals)

    tiles['fold'] = tile_ks
        
    return(tiles,neighbor_dict)

def spatial_block_cv_split(tiles,k=100,degree=3):
    """
    Create spatially-blocked cross validation splits. 
    Includes a built-in buffer layer of tiles between testing and training set. 
    
    param: tiles: pandas geodataframe of spatial blocks
    param: k: number of cross-validation folds
    param: degree: number of degrees of separation between blocks in same fold (can be 1, 2, or 3)
    returns: list of tuples denoting tile indices in train/test set for each split. 
    """
    
    tiles,neighbor_dict = spatial_kfold(tiles,k=k,degree=degree)
    
    splits = []
    
    for k in tiles['fold'].unique():
        
        test_indices = tiles[tiles['fold']==k].index.values
        excluded_tiles = []
        
        for i in test_indices:
            excluded_tiles += neighbor_dict[i]
            
        # Exclude tiles in test set from training
        m1 = np.isin(tiles.index.values,test_indices)
        
        # Also exclude those bordering test set
        m2 = np.isin(tiles.index.values,excluded_tiles)
            

        train_indices = tiles.index.values[~(m1|m2)]
        splits.append((train_indices,test_indices))
        
    return(splits)
    
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
        
        # Adjust damages for inflation
        claims['total_cost'] = claims['total_cost']*inflation_multiplier

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
        claims_filter = claims_filter&(claims['total_cost'] > 0.0)
        payouts_by_building = claims[claims_filter][['building_id','total_cost']].groupby('building_id').max().reset_index()
        flooded_building_ids = payouts_by_building['building_id'].values
        flood_cost = payouts_by_building['total_cost'].values

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
        
        nonflood_cost = np.zeros(nonflooded_building_ids.shape)
        inconclusive_cost = np.ones(inconclusive_building_ids.shape)*np.nan

        event_building_ids = np.concatenate((flooded_building_ids,nonflooded_building_ids,inconclusive_building_ids))
        event_damage_status = np.concatenate((flood_status,nonflood_status,inconclusive_status))
        event_damage_cost = np.concatenate((flood_cost,nonflood_cost,inconclusive_cost))

        # Store properties of flood event
        self.flood_damage_status = pd.DataFrame(data={'building_id':event_building_ids,'flood_damage':event_damage_status,'total_cost':event_damage_cost})
        #self.flood_damage_status = pd.merge(self.flood_damage_status,payouts_by_building,how='left',on='building_id')
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
        auxiliary_policies['policyEffectiveDate'] = pd.to_datetime(auxiliary_policies['policyEffectiveDate'],utc=True).dt.tz_localize(None)
        auxiliary_policies['policyTerminationDate'] = pd.to_datetime(auxiliary_policies['policyTerminationDate'],utc=True).dt.tz_localize(None)

        # Get claims with non-zero payout associated with dates of event
        claims_filter = (auxiliary_claims['dateOfLoss'] >= self.start_date)
        claims_filter = claims_filter&(auxiliary_claims['dateOfLoss'] <= self.end_date)
        claims_filter = claims_filter&(auxiliary_claims['total_cost'] > 0.0)
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
        auxiliary_claims['total_cost'] = auxiliary_claims['total_cost']*inflation_multiplier

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
            pseudo_absences['total_cost'] = 0
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

    def random_cross_validation(self,presence_response_variable,presence_features,cost_response_variable,cost_features,use_adjusted=True,k=5,n_cores=1):
        """
        param: presence_response_variable: name of binary response variable indicating presence/absence of flooding
        param: presence_features: list of features used to predict the presence/absence of flood damage
        param: cost_response_variable: name of continuous variable indicating cost of damages
        param: cost_features: list of features used to predict the cost of damage to flooded structures
        param: use_adjusted: if true, train models using adjusted training data which includes pseudo-absences
        param: k: number of k-fold cross validation iterations to perform
        param: n_cores: number of cores to use if running tasks in parallel
        """
        
        print(f'\n*** {k}-fold cross validation (random) ***',flush=True)

        kf = KFold(n_splits=k,random_state=None,shuffle=True)
        
        if use_adjusted:
            data = self.adjusted_training_dataset
        else:
            data = self.training_dataset
            
        predictions_list = []
        
        t0 = time.time()
            
        for i,(train_indices,test_indices) in enumerate(kf.split(data)):
            
            t1 = time.time()
            
            train_df = data.iloc[train_indices].copy()
            test_df = data.iloc[test_indices].copy()
            
            predictions = cv_fold(i,train_df,test_df,presence_response_variable,presence_features,cost_response_variable,cost_features,n_cores=n_cores)
            predictions_list.append(predictions)
            
            t2 = time.time()
            
            elapsed_time = format_elapsed_time(t2-t1)
            cumulative_elapsed_time = format_elapsed_time(t2-t0)
            
            print(f'CV fold {i+1} / {k} (time elapsed: {elapsed_time} last iteration / {cumulative_elapsed_time} cumulative)',flush=True)

        # Calculate performance metrics
        predictions_df = pd.concat(predictions_list)
        
        y_pred = predictions_df[f'{presence_response_variable}_prob'].to_numpy()
        y_class = predictions_df[f'{presence_response_variable}_class'].to_numpy()
        y_true = predictions_df[presence_response_variable].to_numpy()
        c_pred = predictions_df[f'{cost_response_variable}_pred'].to_numpy()
        c_true = predictions_df[cost_response_variable].to_numpy()
        
        results_dict,roc_curve,pr_curve = performance_metrics(y_pred,y_class,y_true,c_pred,c_true)
        
        self.random_cv_predictions = predictions_df
        self.random_cv_performance_metrics = results_dict
        self.random_cv_roc_curve = roc_curve
        self.random_cv_pr_curve = pr_curve

        return(None)
    
    def spatial_cross_validation(self,presence_response_variable,presence_features,cost_response_variable,cost_features,tiles,use_adjusted=True,max_k=500,n_cores=1):
        """
        param: presence_response_variable: name of binary response variable indicating presence/absence of flooding
        param: presence_features: list of features used to predict the presence/absence of flood damage
        param: cost_response_variable: name of continuous variable indicating cost of damages
        param: cost_features: list of features used to predict the cost of damage to flooded structures
        param: tiles: spatial blocks used to define cross-validation splits
        param: use_adjusted: if true, train models using adjusted training data which includes pseudo-absences
        param: max_k: maximum number of cross-validation folds
        param: n_cores: number of cores to use if running tasks in parallel
        """
        
        tiles = tiles[tiles['geometry'].intersects(self.study_area)][['geometry']]
        tiles.index = np.arange(len(tiles))
        
        if use_adjusted:
            data = self.adjusted_training_dataset
        else:
            data = self.training_dataset
            
        data = gpd.sjoin(data,tiles).rename(columns={'index_right':'tile_index'})
        
        splits = spatial_block_cv_split(tiles,k=max_k)
        k = len(splits)
                
        print(f'\n*** {k}-fold cross validation (spatial) ***',flush=True)
            
        predictions_list = []
        
        t0 = time.time()
            
        for i,(train_tile_indices,test_tile_indices) in enumerate(splits):
            
            t1 = time.time()
            
            train_df = data[data['tile_index'].isin(train_tile_indices)].copy()
            test_df = data[data['tile_index'].isin(test_tile_indices)].copy()
            
            if len(test_df) > 0:
            
                predictions = cv_fold(i,train_df,test_df,presence_response_variable,presence_features,cost_response_variable,cost_features,n_cores=n_cores)
                predictions_list.append(predictions)
            
            t2 = time.time()
            
            elapsed_time = format_elapsed_time(t2-t1)
            cumulative_elapsed_time = format_elapsed_time(t2-t0)
            
            print(f'CV fold {i+1} / {k} (time elapsed: {elapsed_time} last iteration / {cumulative_elapsed_time} cumulative)',flush=True)

        # Calculate performance metrics
        predictions_df = pd.concat(predictions_list)
        
        y_pred = predictions_df[f'{presence_response_variable}_prob'].to_numpy()
        y_class = predictions_df[f'{presence_response_variable}_class'].to_numpy()
        y_true = predictions_df[presence_response_variable].to_numpy()
        c_pred = predictions_df[f'{cost_response_variable}_pred'].to_numpy()
        c_true = predictions_df[cost_response_variable].to_numpy()
        
        results_dict,roc_curve,pr_curve = performance_metrics(y_pred,y_class,y_true,c_pred,c_true)
        
        self.spatial_cv_tiles = tiles
        self.spatial_cv_splits = splits
        self.spatial_cv_predictions = predictions_df
        self.spatial_cv_performance_metrics = results_dict
        self.spatial_cv_roc_curve = roc_curve
        self.spatial_cv_pr_curve = pr_curve

        return(None)
    
    def predict_flood_damage(self,presence_response_variable,presence_features,cost_response_variable,cost_features,use_adjusted=True,n_cores=1):
        """
        param: presence_response_variable: name of binary response variable indicating presence/absence of flooding
        param: presence_features: list of features used to predict the presence/absence of flood damage
        param: cost_response_variable: name of continuous variable indicating cost of damages
        param: cost_features: list of features used to predict the cost of damage to flooded structures
        param: use_adjusted: if true, train models using adjusted training data which includes pseudo-absences
        param: n_cores: number of cores to use if running tasks in parallel
        """
        if use_adjusted:
            train_df = self.adjusted_training_dataset
        else:
            train_df = self.training_dataset
            
        # Determine hyperparameters for random forest models
        presence_hyperparams,cost_hyperparams = tune_hyperparams(train_df,presence_response_variable,presence_features,cost_response_variable,cost_features,n_cores=n_cores)
        
        # Train presence / absence prediction model
        presence_mod = RegressionObject(train_df,train_df,self.target_dataset,presence_response_variable,presence_features,n_cores=n_cores,hyperparams=presence_hyperparams)
        presence_mod.model_fit()
        
        # Determine maximum accuracy threshold based on training data
        y_pred_threshold = presence_mod.model_predict(presence_mod.x)
        y_true_threshold = presence_mod.y
        threshold = maximized_accuracy_threshold(y_pred_threshold,y_true_threshold)
        self.threshold = threshold
        
        # Predict presence of flood damage
        self.target_dataset[f'{presence_response_variable}_prob'] = presence_mod.model_predict(presence_mod.x_target)
        self.target_dataset[f'{presence_response_variable}_class'] = presence_mod.model_classify(presence_mod.x_target,threshold)
        
        # Get feature importance
        importances = presence_mod.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in presence_mod.model.estimators_], axis=0)
        importance_order = np.argsort(importances)

        importances = importances[importance_order]
        std = std[importance_order]
        sorted_features = np.array(presence_features)[importance_order]
        
        self.presence_feature_importance = pd.DataFrame({'feature':sorted_features,'importance':importances,'std':std})
        self.presence_feature_importance = self.presence_feature_importance.sort_values(by='importance',ascending=False).reset_index(drop=True)
        self.presence_response_variable = presence_response_variable
        self.presence_features = presence_features
        self.presence_model = presence_mod
        self.presence_hyperparams = presence_hyperparams
        
        # Predict cost of flood damage
        m = (train_df[presence_response_variable]==1)
        cost_mod = RegressionObject(train_df[m],train_df[m],self.target_dataset,cost_response_variable,cost_features,n_cores=n_cores,hyperparams=cost_hyperparams)
        
        cost_mod.model_fit()
        
        # Predict cost of damage among buildings predicted as flooded
        self.target_dataset[cost_response_variable] = cost_mod.model_predict(cost_mod.x_target)
        
        # Assume buildings classified as non-flooded don't incur damage costs
        self.target_dataset[cost_response_variable] = self.target_dataset[cost_response_variable]*self.target_dataset[f'{presence_response_variable}_class']
        
        # Get feature importance
        importances = cost_mod.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in cost_mod.model.estimators_], axis=0)
        importance_order = np.argsort(importances)

        importances = importances[importance_order]
        std = std[importance_order]
        sorted_features = np.array(cost_features)[importance_order]
        
        self.cost_feature_importance = pd.DataFrame({'feature':sorted_features,'importance':importances,'std':std})
        self.cost_feature_importance = self.cost_feature_importance.sort_values(by='importance',ascending=False).reset_index(drop=True)
        self.cost_response_variable = cost_response_variable
        self.cost_features = cost_features
        self.cost_model = cost_mod
        self.cost_hyperparams=cost_hyperparams
        
        return(None)

    def aggregate_flood_damage(self,stratification_columns,use_adjusted=True):
        """
        Combine estimates of insured (i.e., training) and uninsured (i.e., target) losses to create
        overall estimate of flood damage in study region.

        param: stratification_columns: list of columns used to aggregate damage estimates.
        """
        
        insured_df = self.training_dataset[['building_id','study_area','geometry','flood_damage','total_cost'] + stratification_columns].rename(columns={'flood_damage':'flood_damage_class'})
        insured_df = insured_df[insured_df['study_area']==1]
        insured_df['flood_damage_prob'] = insured_df['flood_damage_class']
        insured_df['insured'] = 1
        uninsured_df = self.target_dataset[['building_id','study_area','geometry','flood_damage_prob','flood_damage_class','total_cost'] + stratification_columns]
        uninsured_df = uninsured_df[uninsured_df['study_area']==1]
        uninsured_df['insured'] = 0
        combined_df = pd.concat([insured_df,uninsured_df]).reset_index(drop=True)
        combined_columns = ['building_id','study_area'] + stratification_columns + ['insured','flood_damage_prob','flood_damage_class','total_cost','geometry']
        combined_df = combined_df[combined_columns]

        agg_dict = {'building_id':'count','flood_damage_class':'sum','total_cost':'sum'}
        insured_rename_dict = {'building_id':'n_insured','flood_damage_class':'n_flooded_insured','total_cost':'cost_flooded_insured'}
        uninsured_rename_dict = {'building_id':'n_uninsured','flood_damage_class':'n_flooded_uninsured','total_cost':'cost_flooded_uninsured'}
        combined_rename_dict = {'building_id':'n_total','flood_damage_class':'n_flooded_total','total_cost':'cost_flooded_total'}

        insured_agg = insured_df[stratification_columns + list(agg_dict.keys())].groupby(stratification_columns).agg(agg_dict).rename(columns=insured_rename_dict)
        uninsured_agg = uninsured_df[stratification_columns + list(agg_dict.keys())].groupby(stratification_columns).agg(agg_dict).rename(columns=uninsured_rename_dict)
        combined_agg = combined_df[stratification_columns + list(agg_dict.keys())].groupby(stratification_columns).agg(agg_dict).rename(columns=combined_rename_dict)

        agg_df = insured_agg.join(uninsured_agg,how='outer').join(combined_agg,how='outer')
        agg_df = agg_df[['n_insured','n_uninsured','n_total','n_flooded_insured','n_flooded_uninsured','n_flooded_total','cost_flooded_insured','cost_flooded_uninsured','cost_flooded_total']].reset_index()
        agg_df = agg_df.sort_values(by=stratification_columns)

        return(agg_df,combined_df)
