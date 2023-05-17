#!/usr/bin/env python
# coding: utf-8

import os
import pwd
import re
import time
import gc
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime as dt
from shapely.geometry import MultiPolygon

def create_gdf(ds, src_prj=None, dst_prj=None, lon_col=None, lat_col=None):
    if src_prj is None:
        src_prj = 4326
        print('Assuming source projection is EPSG 4326.')

    if (lon_col is not None) & (lat_col is not None):
        gdf = gpd.GeoDataFrame(ds,
                               geometry=gpd.points_from_xy(x=ds[lon_col],
                                                           y=ds[lat_col],
                                                           crs=src_prj))
    else:
        matches = [match for match in ds.columns if 'tude' in match]
        if not ds.empty:
            print('Which columns should be used for georeferencing?')
            with pd.option_context('display.max_rows', None, 'display.max_columns',
                                   None):  # more options can be specified also
                print(matches)
            indx_lat = int(input("Enter the index of the lat columns to use: "))
            indx_lon = int(input("Enter the index of the lon columns to use: "))

            gdf = gpd.GeoDataFrame(ds,
                                   geometry=gpd.points_from_xy(x=ds[matches[indx_lon]],
                                                               y=ds[matches[indx_lat]],
                                                               crs=src_prj))
    if dst_prj: gdf.to_crs(crs=dst_prj, inplace=True)

    return gdf


def clean_money_columns(ds):
    for col in ds.columns:
        if col.find('Coverage') > 0 or col.find('Payment') > 0:
            ds[col] = ds[col].replace({'\$': '', '\,': '', '\(': '', '\)': ''}, regex=True)
            ds[col] = ds[col].astype(int)
    return ds


def read_claims_data_as_gdf(file, tstart=None, tend=None):
    ds = pd.read_table(filepath_or_buffer=file,
                       delimiter=None,
                       header='infer',
                       index_col=0)
    ds['Date of Loss'] = pd.to_datetime(ds['Date of Loss'])
    ds = clean_money_columns(ds)
    if tstart:
        tstart = dt.datetime.strptime(tstart, "%m/%d/%Y")
        ds = ds[(ds['Date of Loss'] >= tstart)]
    if tend:
        tend = dt.datetime.strptime(tend, "%m/%d/%Y")
        ds = ds[(ds['Date of Loss'] <= tend)]
    return ds


def read_policy_data_as_gdf(file, tstart=None, tend=None):
    ds = pd.read_table(filepath_or_buffer=file,
                       delimiter=None,
                       header='infer',
                       index_col=0)
    ds['Policy Effective Date'] = pd.to_datetime(ds['Policy Effective Date'])
    ds['Policy Expiration Date'] = pd.to_datetime(ds['Policy Expiration Date'])

    if tstart:
        tstart = dt.datetime.strptime(tstart, "%m/%d/%Y")
        ds = ds[(ds['Policy Effective Date'] <= tstart)]
    if tend:
        tend = dt.datetime.strptime(tend, "%m/%d/%Y")
        ds = ds[(ds['Policy Expiration Date'] >= tend)]
    return ds

def remove_overlaps(gdf):
    """
    param: gdf: geodataframe with overlapping polygon features
    returns: geodataframe where there are no overlaps between polygon features
    """
    for i in gdf.index:

        # Select polygon of interest
        polygon = gdf.loc[i,'geometry']

        # Get indices of other polygons in dataset that intersect polygon
        intersect_bool = gdf.loc[gdf.index != i,'geometry'].intersects(polygon)

        # Get indices of other polygons in dataset that touch (but do not overlap with) polygon
        # These are okay so we don't want to mess with them
        touch_bool = gdf.loc[gdf.index != i,'geometry'].touches(polygon)

        # Get indices of other polygons that overlap or are equivalent in shape to polygon
        problem_bool = (intersect_bool)&(~touch_bool)

        # For clip out overlapping sections
        if problem_bool.sum() > 0:
            problem_ids = problem_bool[problem_bool].index
            gdf.loc[problem_ids,'geometry'] = gdf.loc[problem_ids,'geometry'].difference(polygon)

        return(gdf)

def overlapping_area(x,right,right_id_col):
    """
    Helper function that calculates overlapping area between two polygons match via spatial join
    Function is set up in such a way that works with pandas .apply() methods

    param: x: row of first geodataframe
    param: right_geom: second geodataframe
    param: right_id_col: column in first gdf denoting index of matched polygon in second gdf
    returns: overlapping area between polygon features
    """
    polygon_1 = x['geometry']
    polygon_2 = right.loc[x[right_id_col],'geometry']
    area = polygon_1.intersection(polygon_2).area
    return(area)

def sjoin_polygons_max_overlap(left,right,left_id='index',right_id='index'):
    """
    Spatially join two geodataframes (gdf) of polygon features, retaining keys from the left gdf.
    In the event that a polygon in the left gdf intersects with >1 polygon in the right gdf,
    select the polygon from the right gdf that maximizes the amount of overlapping area.

    param: left: left geodataframe
    param: right: right geodataframe
    param: left_id: name of index in left dataframe (if using named index)
    param: right_id: name of index in right dataframe (if using named index)
    returns: left gdf with column representing matched records in right gdf (equivalent to a sql left join)
    """
    gdf = gpd.sjoin(left,right,'inner').rename(columns={'index_right':right_id})
    gdf['overlapping_area'] = gdf.apply(overlapping_area,axis=1,args=(right,right_id))
    gdf.reset_index(inplace=True)
    gdf.sort_values(by=[left_id,'overlapping_area'],ascending=[True,False],inplace=True)
    df = gdf[[left_id,right_id,'overlapping_area']]
    df = df.groupby(by=left_id).first()
    df.drop(columns=['overlapping_area'],inplace=True)
    return(left.join(df))

def sjoin_points_to_buildings(points,buildings,parcels=None,nmax=None,dmax=None,point_id='claim_id',building_id='building_id',parcel_id='parcel_id'):
    """
    Spatially join points in points geodataframe (gdf) to nearest feature in buildings gdf.
    In the event that a feature in the buildings gdf is matched to >1 point in points gdf,
    retain only the n-nearest points. If a parcels gdf is also passed,
    then any unmatched point that lands on a parcel with exactly one building will be joined
    to that building.

    param: points: geodataframe (point features)
    param: buildings: buildings geodataframe (polygon features). If using parcels, must have a parcel_id column
    param: parcels: parcels geodataframe (polygon features)
    param: nmax: maximum number of points that a feature in buildings gdf can be matched to
    param: dmax: maximum allowable distance between matched features in points and buildings gdfs
    param: point_id: name of index in points dataframe (if using named index)
    param: building_id: name of index in buildings dataframe (if using named index)
    param: parcel_id: name of index in parcels dataframe (if using named index)
    returns: points gdf with column representing matched records in buildings gdf (equivalent to a sql left join)
    """
    smallnum = np.finfo(np.float32).eps # Float tolerance (used to check if )

    n_initial = len(points)

    # Join points to nearest building within search radius
    df = gpd.sjoin_nearest(points,buildings[['geometry']],how='inner',max_distance=dmax,distance_col='distance_to_building').rename(columns={'index_right':building_id})
    df = df[[building_id,'distance_to_building']]

    # If we have data on parcels, attempt to indirectly match unmatched points to building via parcel
    if parcels is not None:
        buildings_per_parcel = buildings.reset_index()[[building_id,parcel_id]].groupby(parcel_id).count()
        one_building_parcel_ids = buildings_per_parcel.index.values[buildings_per_parcel[building_id]==1]
        one_building_parcels = parcels.loc[one_building_parcel_ids,['geometry']]
        one_building_parcels = one_building_parcels.reset_index().merge(buildings.reset_index()[[building_id,parcel_id]],on=parcel_id,how='left')
        remaining_points = points.loc[points.index[~points.index.isin(df.index)],['geometry']]
        indirect_matches = gpd.sjoin(remaining_points,one_building_parcels,how='inner',predicate='within')[[building_id]]
        indirect_matches['distance_to_building'] = np.nan
        n_indirect = len(indirect_matches)
        df = pd.concat([df,indirect_matches])
    else:
        n_indirect = 0

    n_zero = len(df[df['distance_to_building'] <= smallnum])
    n_radius = len(df) - n_zero - n_indirect

    print(f'{n_zero} / {n_initial} ({np.round(n_zero/n_initial*100,2)}%) of points land on a building.',flush=True)

    if dmax is not None:
        print(f'{n_radius} / {n_initial} ({np.round(n_radius/n_initial*100,2)}%) of points are within {max_distance} distance units of a building.',flush=True)
        if parcels is not None:
            print(f'{n_indirect} / {n_initial} ({np.round(n_indirect/n_initial*100,2)}%) of points were indirectly matched to buildings via parcels.',flush=True)

    df.reset_index(inplace=True)

    count = df[[point_id,building_id]].groupby(by=building_id).count()
    n_features = len(count)
    n_one_to_one = count.value_counts().loc[1].values[0]

    print(f'Of matched buildings, {n_one_to_one} / {n_features} ({np.round(n_one_to_one/n_features*100,2)}%) have exactly one point.',flush=True)

    # Sort points based on distance to building
    df.sort_values(by=[building_id,'distance_to_building'],ascending=[True,True],inplace=True)

    # If constraining the number of points per building, keep n nearest points and drop the rest
    if nmax is not None:
        df = df.groupby(by=building_id).nth(np.arange(nmax))
        df.reset_index(inplace=True)
        df.set_index(point_id,inplace=True)

        n_remaining = len(df)

        print(f'After enforcing maximum of {nmax} points per building, {n_remaining} / {n_match} ({np.round(n_remaining/n_match*100,2)}%) of matched points remain.',flush=True)

    else:
        n_remaining = n_match
        df.reset_index(drop=True,inplace=True)
        df.set_index(point_id,inplace=True)

    # Return points gdf with column of ids from matched features in buildings gdf
    return(points.join(df))

def get_one_to_n(left,right,left_id='index',right_id='index'):
    """
    left: dataframe that is in a one-to-n relationship with right df
    right: dataframe that is in a n-to-one relationship with left df
    left_id: named index of left df which occurs as column in right df
    right_id: named index of right df
    returns: left df which now includes a column of lists of matching entries in right df
    """
    left = left.join(right.reset_index().groupby(left_id)[right_id].apply(lambda x: list(x)))
    left[right_id] = left[right_id].apply(lambda x: x if isinstance(x,list) else [])
    return(left)

def fix_column_order(df,cols_to_order):
    new_cols = df.columns.drop(cols_to_order).to_list() + cols_to_order
    return(df[new_cols])

def write_geodatabase(gdf,filepath,layer_name,polygons_as_points=False):
    """
    param: gdf: pandas geodataframe
    param: filepath: filepath to geodatabase
    param: layer_name: name of layer in geodatabase to save file to
    param: polygons_as_points: if true, convert polygon geometries to points based on centroid
    """

    # Convert lists to strings
    column_list = list(gdf.columns)
    for column in column_list:
        if type(gdf[column].iloc[0]) == list:
            gdf[column] = gdf[column].astype(str)

    # If geodataframe contains mix of polygons and multipolygons,
    # upcast polygons to multipolygons

    geom_types = list(gdf['geometry'].geom_type.unique())
    geom_types.sort()

    if polygons_as_points:
        if any(x in geom_types for x in ['Polygon','MultiPolygon']):
            gdf['geometry'] = gdf['geometry'].centroid

    elif len(geom_types) > 1:
        print('warning: multiple geometry types in geodataframe')
        if geom_types == ['MultiPolygon','Polygon']:
            print('upcasting polygons to multipolygons to achieve consistency')
            gdf["geometry"] = [MultiPolygon([feature]) if feature.geom_type == 'Polygon' else feature for feature in gdf['geometry']]

    gdf.to_file(filepath,layer=layer_name,driver='OpenFileGDB')

    return(None)

# Get information on user profile and date of script execution
username = pwd.getpwuid(os.getuid())[0]
current_datetime = dt.datetime.today().strftime('%Y-%m-%d %H:%M:%S')

# Specify path to files

pwd = os.getcwd()

parcels_filepath = '/proj/characklab/flooddata/NC/NC_statewide_buildings_parcels/NC_parcels_all.gdb'
buildings_filepath = '/proj/characklab/flooddata/NC/NC_statewide_buildings_parcels/NC_Buildings_p.gdb'
claims_filepath = '/proj/characklab/flooddata/NC/geocoding/geocoded_datasets/NFIP_claim_data_geocoded_merged.txt'
policies_filepath = '/proj/characklab/flooddata/NC/geocoding/geocoded_datasets/NFIP_policy_data_geocoded_merged.txt'

# Create folders for output
today = dt.datetime.today().strftime('%Y-%m-%d')
outfolder = os.path.join(pwd,f'{today}_joined_data')

print('\n\n*** Run Information ***\n\n',flush=True)
print(f'User: {username}',flush=True)
print(f'Date: {current_datetime}',flush=True)
print(f'Working directory: {pwd}',flush=True)
print(f'Output directory: {outfolder}',flush=True)
print(f'Parcels data source: {parcels_filepath}',flush=True)
print(f'Buildings data source: {buildings_filepath}',flush=True)
print(f'Claims data source: {claims_filepath}',flush=True)
print(f'Policies data source: {policies_filepath}',flush=True)

if not os.path.exists(outfolder):
    os.makedirs(outfolder)

# Make everything UTM 17N so that we can compute distances
epsg_code = 32617

print('\n\n*** Read and Reproject Data ***\n\n',flush=True)
t0 = time.time()

print(f'Target coordinate reference system: EPSG:{epsg_code}',flush=True)

# Read in claims
claims = read_claims_data_as_gdf(file=claims_filepath)

# Create geodataframe (reproject if necessary), specify the lat, lon column

# For rooftop-matched claims, use google lat/lon (WGS84)
rooftop_claims_gdf = create_gdf(claims[claims['google_location_type']=='ROOFTOP'],
                                src_prj=4326,
                                dst_prj=epsg_code,
                                lon_col='google_longitude',
                                lat_col='google_latitude')

# For non-rooftop-matched claims, use FEMA lat/lon (NAD83)
nonrooftop_claims_gdf = create_gdf(claims[claims['google_location_type']!='ROOFTOP'],
                                src_prj=4269,
                                dst_prj=epsg_code,
                                lon_col='Latitude',
                                lat_col='Longitude')

# Combine rooftop and non-rooftop claims into single dataset in specified projected coordinate system
claims_gdf = pd.concat([rooftop_claims_gdf,nonrooftop_claims_gdf])

# Read in policies
policies = read_policy_data_as_gdf(file=policies_filepath)

# Create geodataframe (reproject if necessary), specify the lat, lon column
policies_gdf = create_gdf(policies,
                          src_prj=4326,
                          dst_prj=epsg_code,
                          lon_col='google_longitude',
                          lat_col='google_latitude')

# Read in buildings data
buildings_gdf = gpd.read_file(buildings_filepath,layer='NC_Buildings')
buildings_gdf.to_crs(epsg=epsg_code, inplace=True)

# Read in parcels data
parcels_gdf = gpd.read_file(parcels_filepath,layer='nc_parcels_poly')
parcels_gdf.to_crs(epsg=epsg_code, inplace=True)


# Create unique ID for each building and parcel

# First reset indices to ensure they're valid
buildings_gdf.reset_index(drop=True,inplace=True)
parcels_gdf.reset_index(drop=True,inplace=True)

buildings_gdf['building_id'] = buildings_gdf.index.values
parcels_gdf['parcel_id'] = parcels_gdf.index.values

buildings_gdf.set_index('building_id',inplace=True)
parcels_gdf.set_index('parcel_id',inplace=True)

# Ensure that here are no overlapping sections of buildings
buildings_gdf = remove_overlaps(buildings_gdf)

# Ensure that there are no overlapping sections of parcels
parcels_gdf = remove_overlaps(parcels_gdf)

t1 = time.time()
print(f'Time elapsed: {dt.timedelta(seconds=int(t1-t0))}',flush=True)
print(f'Cumulative time elapsed: {dt.timedelta(seconds=int(t1-t0))}',flush=True)
gc.collect()

# Join buildings to parcels
# Drop any building that does not match to a parcel

print('\n\n*** Buildings to Parcels ***\n\n',flush=True)

buildings_gdf = sjoin_polygons_max_overlap(buildings_gdf,parcels_gdf,left_id='building_id',right_id='parcel_id')
n_total = len(buildings_gdf)
n_missing = buildings_gdf['parcel_id'].isna().sum()
n_match = n_total - n_missing

print(f'Matched {n_match} / {n_total} ({np.round(n_match/n_total*100,2)}%) of buildings to a parcel.',flush=True)

# Drop buildings that didn't match to a parcel
excluded_buildings_gdf = buildings_gdf[buildings_gdf['parcel_id'].isna()]
buildings_gdf = buildings_gdf[~buildings_gdf['parcel_id'].isna()]
buildings_gdf['parcel_id'] = buildings_gdf['parcel_id'].astype('int')


# Create unique ID for each claim and policy

# First drop any duplicate entries to ensure each row is unique
claims_gdf.drop_duplicates(inplace=True)
policies_gdf.drop_duplicates(inplace=True)

# Then reset index to ensure indices are valid
claims_gdf.reset_index(drop=True,inplace=True)
policies_gdf.reset_index(drop=True,inplace=True)

claims_gdf['claim_id'] = claims_gdf.index.values
policies_gdf['policy_id'] = policies_gdf.index.values

claims_gdf.set_index('claim_id',inplace=True)
policies_gdf.set_index('policy_id',inplace=True)

t2 = time.time()
print(f'Time elapsed: {dt.timedelta(seconds=int(t2-t1))}',flush=True)
print(f'Cumulative time elapsed: {dt.timedelta(seconds=int(t2-t0))}',flush=True)
gc.collect()

# Join claims to nearest building within search radius
print('\n\n*** Claims to Buildings ***\n\n',flush=True)

# Search radius to used when joining to nearest building
# (should be in units of meters if using UTM)
max_distance=30

# Maximum number of claims allowed per building
# If you want a 1:1 relationship, set equal to 1
# If you do not want to place a constraint on this, set equal to None
max_claims_per_building=None

claims_gdf = sjoin_points_to_buildings(claims_gdf,buildings_gdf,parcels=parcels_gdf,nmax=max_claims_per_building,dmax=max_distance,point_id='claim_id',building_id='building_id',parcel_id='parcel_id')

# Drop claims that didn't match to a building
excluded_claims_gdf = claims_gdf[claims_gdf['building_id'].isna()]
claims_gdf = claims_gdf[~claims_gdf['building_id'].isna()]
claims_gdf['building_id'] = claims_gdf['building_id'].astype('int')

n_included = len(claims_gdf)
n_excluded = len(excluded_claims_gdf)
n_initial = n_included + n_excluded

print(f'Final dataset consists of {n_included} claims ({np.round(n_included/n_initial*100,2)}% of original dataset.)',flush=True)

t3 = time.time()
print(f'Time elapsed: {dt.timedelta(seconds=int(t3-t2))}',flush=True)
print(f'Cumulative time elapsed: {dt.timedelta(seconds=int(t3-t0))}',flush=True)
gc.collect()

# Policies to buildings

print('\n\n*** Policies to Buildings ***\n\n',flush=True)

n_initial = len(policies_gdf)

# Drop any policies that aren't rooftop matches in google geocoding api
# These are typically only geolocated to a city or town, or the middle of a street at best
# For claims, this isn't as much of a problem because we have fema lat/lon as a backup,
# but for policies we have no alternative so it's probably safest to drop.

excluded_policies_gdf = policies_gdf[policies_gdf['google_location_type']!='ROOFTOP']
policies_gdf = policies_gdf[policies_gdf['google_location_type']=='ROOFTOP']

n_rooftop = len(policies_gdf)
n_nonrooftop = n_initial - n_rooftop

print(f'{n_rooftop} / {n_initial} ({np.round(n_rooftop/n_initial*100,2)}%) of policies were geolocated to a specific building in google maps api.',flush=True)

# Maximum number of policies allowed per building
# If you want a 1:1 relationship, set equal to 1
# If you do not want to place a constraint on this, set equal to None
# (in most cases you probably want this to be None)
max_policies_per_building=None

# Join policies to nearest building within search radius
policies_gdf = sjoin_points_to_buildings(policies_gdf,buildings_gdf,parcels=parcels_gdf,nmax=max_policies_per_building,dmax=max_distance,point_id='policy_id',building_id='building_id',parcel_id='parcel_id')

# Drop policies that didn't match to a building
excluded_policies_gdf = pd.concat([excluded_policies_gdf,policies_gdf[policies_gdf['building_id'].isna()]])
policies_gdf = policies_gdf[~policies_gdf['building_id'].isna()]
policies_gdf['building_id'] = policies_gdf['building_id'].astype('int')

n_included = len(policies_gdf)
n_excluded = len(excluded_policies_gdf)

print(f'Final dataset consists of {n_included} policies ({np.round(n_included/n_initial*100,2)}% of original dataset.)',flush=True)

t4 = time.time()
print(f'Time elapsed: {dt.timedelta(seconds=int(t4-t3))}',flush=True)
print(f'Cumulative time elapsed: {dt.timedelta(seconds=int(t4-t0))}',flush=True)
gc.collect()

# Claims to policies

print('\n\n*** Claims to Policies ***\n\n',flush=True)

# In theory, we would expect each claim to be associated with one unique policy;
# however, because we sometimes have multiple overlapping policies that match to the same building,
# it ends up being a many-to-many relationship in a small percentage of cases.
# Anecdotally, it seems like this mostly happens when you have a lot of sub-premises clustered
# nearby in space (e.g., buildings A and B in an apartment complex; building B might end up grabbing both policies)

# Link claims to policies based on building_id
c2p = claims_gdf.reset_index().merge(policies_gdf[['building_id','Policy Effective Date','Policy Expiration Date']].reset_index(),on='building_id',how='inner')

# Only keep matches where the claim occurs between start/end dates of policy
c2p = c2p[(c2p['Date of Loss']>=c2p['Policy Effective Date'])&(c2p['Date of Loss']<=c2p['Policy Expiration Date'])]
c2p = c2p[['claim_id','policy_id']]

# Make a copy that we can use to create similar column in policies dataframe
p2c = c2p.copy()

# For each claim, make a list of matched policies. In most cases the length of this list will be 1
c2p = c2p.groupby('claim_id')['policy_id'].apply(lambda x: list(x)).reset_index()
c2p['n_policies'] = c2p['policy_id'].apply(lambda x: len(x))

# For each policy, make a list of matched claims. There's no reason
p2c = p2c.groupby('policy_id')['claim_id'].apply(lambda x: list(x)).reset_index()
p2c['n_claims'] = p2c['claim_id'].apply(lambda x: len(x))

# Join back to claims and policies geodataframes
c2p.set_index('claim_id',inplace=True)
p2c.set_index('policy_id',inplace=True)

claims_gdf = claims_gdf.join(c2p)
policies_gdf = policies_gdf.join(p2c)

claims_gdf['policy_id'] = claims_gdf['policy_id'].apply(lambda x: x if isinstance(x,list) else [])
claims_gdf['n_policies'] = claims_gdf['n_policies'].fillna(value=0).astype(int)

policies_gdf['claim_id'] = policies_gdf['claim_id'].apply(lambda x: x if isinstance(x,list) else [])
policies_gdf['n_claims'] = policies_gdf['n_claims'].fillna(value=0).astype(int)

# Print results

v_counts = claims_gdf['n_policies'].value_counts()
n_total = v_counts.sum()
n_zero = v_counts[0]
n_one = v_counts[1]
n_2plus = n_total - (n_zero + n_one)
n_match = n_one + n_2plus

print(f'{n_match} / {n_total} ({np.round(n_match/n_total*100,2)}%) of claims match to one or more policies.',flush=True)
print(f'Of matched claims, {n_one} / {n_match} ({np.round(n_one/n_match*100,2)}%) match to exactly one policy.',flush=True)

v_counts = policies_gdf['n_claims'].value_counts()
n_total = v_counts.sum()
n_zero = v_counts[0]
n_one = v_counts[1]
n_2plus = n_total - (n_zero + n_one)
n_match = n_one + n_2plus

print(' ',flush=True)
print(f'{n_match} / {n_total} ({np.round(n_match/n_total*100,2)}%) of policies match to one or more claims.',flush=True)
print(f'Of matched policies, {n_one} / {n_match} ({np.round(n_one/n_match*100,2)}%) match to exactly one claim.',flush=True)

# Check the extent to which the google formatted addresses agree among rooftop matched claims
# that are linked to exactly one policy

addr_df = c2p[c2p['n_policies']==1].copy()
addr_df['policy_id'] = addr_df['policy_id'].apply(lambda x: x[0])
addr_df = addr_df.reset_index()
addr_df = addr_df.merge(claims_gdf[['google_location_type','google_formatted_address']].reset_index(),on='claim_id')
addr_df = addr_df[addr_df['google_location_type']=='ROOFTOP']
addr_df = addr_df.merge(policies_gdf[['google_formatted_address']].reset_index(),on='policy_id')
addr_df['address_agreement'] = (addr_df['google_formatted_address_x']==addr_df['google_formatted_address_y'])
addr_df['address_agreement'].value_counts(normalize=True)
n_agree = addr_df['address_agreement'].sum()
n_total = len(addr_df)

print(' ',flush=True)
print(f'Among rooftop-matched claims linked to exactly one policy, {n_agree} / {n_total} ({np.round(n_agree/n_total*100,2)}%) have the same google-formatted address as the policy.',flush=True)

t5 = time.time()
print(f'Time elapsed: {dt.timedelta(seconds=int(t5-t4))}',flush=True)
print(f'Cumulative time elapsed: {dt.timedelta(seconds=int(t5-t0))}',flush=True)
gc.collect()

print('\n\n*** Write Output ***\n\n',flush=True)

# Drop n_claims and n_policies columns
# It's easy to re-calculate these when needed, and including them would make the dataframes
# really bloated when we tie things together at the end
claims_gdf.drop(columns='n_policies',inplace=True)
policies_gdf.drop(columns='n_claims',inplace=True)

# Add claim and policy ids to buildings gdf
buildings_gdf = get_one_to_n(buildings_gdf,claims_gdf,left_id='building_id',right_id='claim_id')
buildings_gdf = get_one_to_n(buildings_gdf,policies_gdf,left_id='building_id',right_id='policy_id')

# Add parcel_id to claims and policies gdfs
claims_gdf = claims_gdf.reset_index().merge(buildings_gdf[['parcel_id']].reset_index(),on='building_id',how='left').set_index('claim_id')
policies_gdf = policies_gdf.reset_index().merge(buildings_gdf[['parcel_id']].reset_index(),on='building_id',how='left').set_index('policy_id')

# Add building, claim and policy ids to parcels gdf
parcels_gdf = get_one_to_n(parcels_gdf,buildings_gdf,left_id='parcel_id',right_id='building_id')
parcels_gdf = get_one_to_n(parcels_gdf,claims_gdf,left_id='parcel_id',right_id='claim_id')
parcels_gdf = get_one_to_n(parcels_gdf,policies_gdf,left_id='parcel_id',right_id='policy_id')

# Fix ordering of columns
parcels_gdf = fix_column_order(parcels_gdf,['building_id','claim_id','policy_id','geometry'])
buildings_gdf = fix_column_order(buildings_gdf,['parcel_id','claim_id','policy_id','geometry'])
claims_gdf = fix_column_order(claims_gdf,['distance_to_building','parcel_id','building_id','policy_id','geometry'])
policies_gdf = fix_column_order(policies_gdf,['distance_to_building','parcel_id','building_id','claim_id','geometry'])

# For excluded buildings, claims and policies, add column explaining why it was excluded
excluded_buildings_gdf['reason_excluded'] = 'Buliding does not land on a parcel.'
excluded_claims_gdf['reason_excluded'] = f'Claim is >{max_distance} distance units from nearest building and cannot be indirectly matched to a unique structure via parcel.'
excluded_claims_gdf = excluded_claims_gdf.drop(columns='building_id')
excluded_policies_gdf['reason_excluded'] = f'Policy is >{max_distance} distance units from nearest building and cannot be indirectly matched to a unique structure via parcel.'
excluded_policies_gdf = excluded_policies_gdf.drop(columns='building_id')
excluded_policies_gdf.loc[excluded_policies_gdf['google_location_type'] != 'ROOFTOP','reason_excluded'] = 'Policy could not be geocoded to a specific address; resulting coordinates are unlikely to reflect true location.'

# Save results
included_gdb = os.path.join(outfolder,'included_data.gdb')
excluded_gdb = os.path.join(outfolder,'excluded_data.gdb')

# Included features
write_geodatabase(parcels_gdf,included_gdb,'parcels')
write_geodatabase(buildings_gdf,included_gdb,'buildings',polygons_as_points=True)
write_geodatabase(claims_gdf,included_gdb,'claims')
write_geodatabase(policies_gdf,included_gdb,'policies')

# Excluded features
write_geodatabase(excluded_buildings_gdf,excluded_gdb,'buildings',polygons_as_points=True)
write_geodatabase(excluded_claims_gdf,excluded_gdb,'claims')
write_geodatabase(excluded_policies_gdf,excluded_gdb,'policies')

t6 = time.time()
print(f'Time elapsed: {dt.timedelta(seconds=int(t6-t5))}',flush=True)
print(f'Cumulative time elapsed: {dt.timedelta(seconds=int(t6-t0))}',flush=True)
