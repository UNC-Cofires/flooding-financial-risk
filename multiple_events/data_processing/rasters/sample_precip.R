# Read in libraries

library(ggplot2)
library(sf)
library(dplyr)
library(tmap)
library(raster)


# Specify working directory and file paths
base_dir <- "/proj/characklab/flooddata/NC"
output_dir <-paste(base_dir,"multiple_events/data_processing/rasters",sep="/")
raster_dir <- paste(base_dir,"multiple_events/geospatial_data/daymet/max_3day_precip",sep="/")
buildings_filepath <- paste(base_dir,"data_joining/2023-02-27_joined_data/included_data.gdb",sep="/")

# Specify list of rasters to include
included_rasters <- c("max_3day_precip_1996-09-02_1996-09-17_Fran.tif",
                      "max_3day_precip_1998-08-22_1998-08-31_Bonnie.tif",
                      "max_3day_precip_1999-09-08_1999-09-26_Floyd.tif",
                      "max_3day_precip_2003-09-16_2003-09-24_Isabel.tif",
                      "max_3day_precip_2011-08-24_2011-09-10_Irene.tif",
                      "max_3day_precip_2016-10-05_2016-10-17_Matthew.tif",
                      "max_3day_precip_2018-09-08_2018-09-27_Florence.tif")

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
write.csv(sampled_values,file=paste(output_dir,"max_3day_precip_at_building_points.csv",sep="/"),row.names=FALSE)