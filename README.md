The data used for this repo comes from the https://www.geoportal.nrw/

The feature names used are: 

DGM - Digitales Geländemodell - Rasterweite 1 m (GeoTIFF) - Paketierung: Einzelkacheln

Footprint - Hausumringe (Shape) - Paketierung: gesamt NRW

Point Clouds - 3D-Messdaten Laserscanning (LAS) - Paketierung: Einzelkacheln



Since the footprints are only available for all of NRW, the shapefile_point_cloud_utils has a function crop_point_cloud_boundary_from_huge_shapefile to just crop out the AoI from them first. 
If only a connector for point cloud, shapefile --> a list of dicts with keys: polygons, height is required use get_height_from_polygons 
