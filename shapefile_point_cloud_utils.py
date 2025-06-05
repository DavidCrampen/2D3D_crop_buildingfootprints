import laspy
import numpy as np
from shapely.geometry import box
from shapely.geometry import Point, Polygon, mapping
from tqdm import tqdm
import geopandas as gpd
import os
import pandas as pd
import json
import torch

def is_points_inside_polygon(point_cloud_2d, polygon):
    # Convert polygon to tensors on GPU
    polygon = torch.tensor(polygon, dtype=torch.float32, device=point_cloud_2d.device)

    # Get the number of points in the polygon
    n = len(polygon)

    # Shift the polygon so that the first point is repeated to close the loop
    polygon = torch.cat([polygon, polygon[:1]], dim=0)

    # Extract x and y coordinates of the polygon and the point cloud
    x, y = point_cloud_2d[:, 0], point_cloud_2d[:, 1]
    polygon_x, polygon_y = polygon[:, 0], polygon[:, 1]

    # Compute the vectorized ray-casting method
    # Using `torch` for element-wise comparison and operations
    inside = torch.zeros(point_cloud_2d.shape[0], dtype=torch.bool, device=point_cloud_2d.device)

    for i in range(n):
        p1x, p1y = polygon_x[i], polygon_y[i]
        p2x, p2y = polygon_x[i + 1], polygon_y[i + 1]

        # Check if the point's y-coordinate is between the y-values of the segment
        cond1 = (y > torch.min(p1y, p2y)) & (y <= torch.max(p1y, p2y))

        # Check if the point's x-coordinate is to the left of the segment's x-intersection
        cond2 = (x <= torch.max(p1x, p2x))

        # Compute the intersection point if the segment is not horizontal
        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x

        # Final condition for being inside the polygon
        cond3 = (p1y != p2y) & (x <= xinters)

        # XOR of conditions to determine if the point is inside
        inside ^= cond1 & cond2 & (cond3 | (p1x == p2x))

    return inside

def crop_point_cloud_torch(point_cloud , polygons, device ="cpu"):

    # Move point cloud to GPU
    point_cloud = torch.tensor(point_cloud, dtype=torch.float32, device=device)

    # Generate polygons

    results = []
    for  polygon in tqdm(polygons, desc="Processing footprints"):
        polygon_coords = torch.tensor(np.array(polygon.exterior.coords), dtype=torch.float32, device=device)

        # Use vectorized point-in-polygon test
        mask = is_points_inside_polygon(point_cloud[:, :2], polygon_coords)
        part = point_cloud[mask]
        if len(part)>0:
            height = torch.max(part[:, 2]).item()

            results.append({
                "height": height,
                "polygon": mapping(polygon)  # GeoJSON format
            })
    return results


def load_shapefiles(filenames, shapefile_path, bbox=None):
    """
    Load one or multiple shapefiles from the given path and return a combined GeoDataFrame.

    Parameters:
    - filenames (str or list of str): Filename or list of filenames to load.
    - shapefile_path (str): Path to the shapefiles.

    Returns:
    - GeoDataFrame: Combined GeoDataFrame containing all loaded geometries.
    """
    if isinstance(filenames, str):
        filenames = [filenames]

    geodataframes = []
    for filename in filenames:
        full_path = os.path.join(shapefile_path, filename)
        print(f"Loading: {full_path}")
        try:
            shp = gpd.read_file(
                full_path,
                engine="pyogrio",
                use_arrow=True,
                bbox=bbox
            )
            print(shp)
            geodataframes.append(shp)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if geodataframes:
        combined_gdf = gpd.GeoDataFrame(pd.concat(geodataframes, ignore_index=True))
        return combined_gdf
    else:
        return None  # No valid shapefiles loaded

def load_point_cloud(point_cloud_name, pc_path, mode= "Photogrammetry", filetype= "las"):
    full_path = os.path.join(pc_path, point_cloud_name)
    if filetype == "las":

        las = laspy.read(full_path)
        if mode == "ALS":
            point_cloud = np.vstack((las.x, las.y,las.z, las.intensity,   las.classification)).T
            b_col= np.zeros((len(point_cloud), 3))
            point_cloud= np.hstack((point_cloud[:,:3], b_col, point_cloud[:,3:]))
        elif mode == "Photogrammetry":
            point_cloud = np.vstack((las.x, las.y, las.z,las.red, las.green, las.blue, las.intensity, las.classification)).T
        elif mode == "UavPhoto":
            point_cloud = np.vstack((las.x, las.y, las.z,las.red, las.green, las.blue, las.classification)).T
        else:
            point_cloud = np.vstack((las.x, las.y, las.z)).T
    elif filetype == "txt":
        point_cloud = np.loadtxt(full_path, delimiter=",")
        las = None
    elif filetype =="npy":
        point_cloud = np.load(full_path)
        las= None

    point_cloud[:,0]+=32000000
    return point_cloud, las

def get_point_cloud_boundary(point_cloud1):
    """
    this funtion implements three types of boundary extraction for an input point cloud, boundary is the unoriented bounding box,
    hull creates a convex hull boundary and prime creates an oriented bounding box rotated to the point clouds principle direction
    :param point_cloud_name: give the point cloud name
    :param pc_path: pass the point cloud directory
    :param mode: Hull generates a convex hull around the point cloud in its xy-dimensions
                Boundary generates the minimum unoriented boundingbox around it
                Prime_boundary generates an oriented bounding box around it
    :return: boundary polygon
    """
    idx = np.random.choice(len(point_cloud1), 10000)
    point_cloud = point_cloud1[idx, :2].copy()

    x_max, x_min = np.max(point_cloud[:, 0]), np.min(point_cloud[:, 0])
    y_max, y_min = np.max(point_cloud[:, 1]), np.min(point_cloud[:, 1])
    boundary_points = np.array([[x_max, y_max], [x_min, y_max], [x_min, y_min], [x_max, y_min]])
    poly = Polygon(boundary_points)


    return poly

def get_point_cloud_bbox(point_cloud):
    """
    Erstellt ein rechteckiges Polygon (Bounding Box) aus den XY-Koordinaten einer Punktwolke.

    Parameters:
    - point_cloud (np.ndarray): Ein Numpy-Array mit mindestens zwei Spalten für X und Y.

    Returns:
    - shapely.geometry.Polygon: Ein Polygonobjekt, das die Bounding Box darstellt.
    """
    min_x, min_y = point_cloud[:, 0].min(), point_cloud[:, 1].min()
    max_x, max_y = point_cloud[:, 0].max(), point_cloud[:, 1].max()
    return box(min_x, min_y, max_x, max_y)
def crop_shapefile_from_polygon(shapefile_gdf, polygon):
    # Clip the shapefile using the polygon
    clipped_gdf = gpd.clip(shapefile_gdf, polygon)
    return clipped_gdf


def extract_polygons_from_shapefile(shapefile_gdf: gpd.GeoDataFrame):
    """
    Extracts only Polygon and MultiPolygon geometries from a GeoDataFrame.

    Parameters:
    - shapefile_gdf (GeoDataFrame): A loaded shapefile as a GeoDataFrame.

    Returns:
    - List of Polygon geometries.
    """
    polygons = []
    for geom in shapefile_gdf.geometry:
        if isinstance(geom, Polygon):
            polygons.append(geom)
        # we are skipping multi poligons, since they make cropping a bit harder
        #elif isinstance(geom, MultiPolygon):
            #polygons.extend(geom.geoms)  # unpack MultiPolygon
    return polygons
def extract_points_from_polygon(point_cloud, polygon):
    """
    This function first performs a coarse cropping inside the boundaries and
    in a second step performs the fine crop with the shapely lib polygon.contains() function

    :param point_cloud: input point cloud to be cropped
    :param polygon: input polygon as boundary for the cropping
    :return: cropped point cloud
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    within_bounds = (
            (point_cloud[:, 0] >= min_x) & (point_cloud[:, 0] <= max_x) &
            (point_cloud[:, 1] >= min_y) & (point_cloud[:, 1] <= max_y)
    )

    filtered_points = point_cloud[within_bounds]
    if len(filtered_points) == 0:
        print("No Points within bound found!")
        return None

    xy_points = filtered_points[:, :2]

    shapely_points = np.array([Point(xy) for xy in xy_points])

    # Step 4: Check which points are inside the polygon
    points_inside_mask = np.array([polygon.contains(point) for point in shapely_points])

    # Final points inside the polygon
    points_in_polygon = filtered_points[points_inside_mask]
    height =np.max(points_in_polygon[:,2])

    return height, points_in_polygon, polygon

def save_shapefile(gdf, filename, output_path):
    full_path= os.path.join(output_path, filename)
    gdf.to_file(full_path)

def crop_point_cloud_boundary_from_huge_shapefile(pc_path, pc_name, shapefile_path, shapefile_name, output_path, out_name):


    test_point_cloud, las = load_point_cloud(pc_name, pc_path, mode="ALS")
    point_cloud_bbox = get_point_cloud_bbox(test_point_cloud)
    shapefile = load_shapefiles(filenames=shapefile_name, shapefile_path=shapefile_path, bbox=point_cloud_bbox)
    aoi_footprints = gpd.clip(shapefile, point_cloud_bbox)
    save_shapefile(aoi_footprints, filename=out_name, output_path=output_path)
    print("Saved cropped shp")


def get_heights_from_polygons(point_cloud, shapefile):
    """This function is for QGIS internal use
    Input: point cloud and shapefile
    Output: json with heights and polygons """
    point_cloud_bbox = get_point_cloud_bbox(point_cloud)
    clipped_shapefile = gpd.clip(shapefile, point_cloud_bbox)
    footprint_polygons = extract_polygons_from_shapefile( clipped_shapefile)
    results = crop_point_cloud_torch(point_cloud, footprint_polygons, device=device)
    """
    results is a list of dicts with keys: height, polygon; for all building footprint available within the point_cloud boundaries"""
    return results


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pc_path= os.path.join(base_dir, "point_clouds")
    shapefile_path =os.path.join(base_dir, "shapefiles")
    output_path =os.path.join(base_dir,"json_outputs")

    test_point_cloud, las= load_point_cloud("3dm_32_293_5629_1_nw.laz", pc_path, mode="ALS")
    point_cloud_bbox = get_point_cloud_bbox(test_point_cloud)

    shapefile = load_shapefiles(filenames ="cropped_footprints.shp",shapefile_path=shapefile_path,bbox=point_cloud_bbox)
    footprint_polygons = extract_polygons_from_shapefile(shapefile)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = crop_point_cloud_torch(test_point_cloud , footprint_polygons, device =device)
    """
    #ohne py torch (langsamer) 
    results = []
    for footprint in tqdm(footprint_polygons, desc="Processing footprints"):
        height, segment, polygon = extract_points_from_polygon(test_point_cloud, footprint)

        results.append({
            "height": height,
            "polygon": mapping(polygon)  # GeoJSON format
        })
    """
    with open(os.path.join(output_path, "polygon_data.json"), "w") as f:
        json.dump(results, f, indent=2)
