import geopandas as gpd
from shapely.geometry import Point


### Checking Schools 
# gdf = gpd.read_file("pitt_schools.shp")
# print(gdf['county'].unique())

# gdf = gpd.read_file(johnston_schools.shp")
# print(gdf['county'].unique())

# gdf = gpd.read_file("all_nc_schools.shp")
# print(gdf['county'].unique())


### Johnston & Pitt Counties Parcel Data is Good
# gdf = gpd.read_file("johnston_parcels.shp")
# print(gdf['CNTYNAME'].unique())

### Pitt County Address Data is Good
gdf = gpd.read_file("pitt_addresses.shp")
# print(gdf['county_nam'].unique())


### Johnston County Address Data is Good
# gdf = gpd.read_file("johnston_addresses.shp")
# print(gdf['County'].unique())

print(gdf.columns)