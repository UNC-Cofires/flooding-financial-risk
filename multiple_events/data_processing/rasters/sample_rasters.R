# Read in libraries

library(ggplot2)
library(sf)
library(dplyr)
library(tmap)
library(raster)



# Specify working directory and file paths
base_dir <- "/proj/characklab/flooddata/NC"
output_dir <-paste(base_dir,"multiple_events/data_processing/rasters",sep="/")
raster_dir <- paste(base_dir,"R_files/FinalRasters",sep="/")
buildings_filepath <- paste(base_dir,"data_joining/2023-02-27_joined_data/included_data.gdb",sep="/")

# Specify list of rasters to include
included_rasters <- c("DistanceCoast/NHDcoastline_DistRaster_500res_08222022.tif",
                      "DistanceRivers/NC_MajorHydro_DistRaster_500res_08292022.tif",
                      "HAND/HANDraster_MosaicR_IDW30_finalR_03032023.tif",
                      "TWI/TWIrasterHuc12_10262022.tif",
                      "Soil/soilsKsat_NC_03072023.tif",
                      "NED/NEDavgslope_NCcrop_huc12_10262022.tif",
                      "NED/NEDraster_resample_07042022.tif",
                      "NLCD_impervious/NLCDimpraster_NC2016_07132022.tif",
                      "SFHA/NC_SFHA_NoX_extend_10252022.tif")

included_rasters <- paste(raster_dir,included_rasters,sep="/")

# Read in rasters as multidimensional array
raster_stack <- stack(included_rasters)
raster_crs <- crs(raster_stack)

# Read in building points
buildings <- read_sf(buildings_filepath,layer="buildings")

# Convert to same CRS as raster stack
buildings <- st_transform(buildings,crs=raster_crs)

# Sample raster values at building points
sampled_values <- extract(raster_stack,buildings,method="simple",df=TRUE)
sampled_values$building_id <- buildings$building_id

# Drop columns we don't need
drops <- c("ID")
sampled_values <- sampled_values[,!(names(sampled_values) %in% drops)]
sampled_values <- sampled_values %>% relocate(building_id)

# Save as CSV
write.csv(sampled_values,file=paste(output_dir,"raster_values_at_building_points.csv",sep="/"),row.names=FALSE)
