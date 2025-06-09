import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

### *** INITIAL SETUP *** ###

pwd = os.getcwd()
sobol_dir = os.path.join(pwd,'sobol_indices')

### *** READ IN BORROWER-LEVEL DATA ON SOBOL INDICES *** ###

subfolders = list([os.path.join(sobol_dir,x) for x in os.listdir(sobol_dir)])
subfolders = np.sort([x for x in subfolders if os.path.isdir(x)])
counties = [x.split('/')[-1] for x in subfolders]
filepaths = [os.path.join(sobol_dir,f'{county}/{county}_sobol_indices.parquet') for county in counties]
filepaths = [x for x in filepaths if os.path.isfile(x)]

sobol_df = pd.concat([pd.read_parquet(f) for f in filepaths]).reset_index(drop=True)

# Save to file
outname = os.path.join(sobol_dir,'sobol_indices_concatenated.parquet')
sobol_df.to_parquet(outname)

### *** AGGREGATE INDICES ACROSS BORROWERS *** ###

# Weight by variance in output
sobol_df['weight'] = sobol_df['output_std']**2

weighted_average = lambda x: np.average(x, weights=sobol_df.loc[x.index,"weight"])

agg_df = sobol_df.groupby(['input_name','output_name']).agg({'sobol_first_order':weighted_average, 'sobol_total_order':weighted_average}).reset_index()

outname = os.path.join(sobol_dir,'sobol_indices_aggregated.parquet')
agg_df.to_parquet(outname)