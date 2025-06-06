import trimesh
from shapely.geometry import shape, Polygon
import numpy as np
import os
import json


import rasterio
from shapely.geometry import mapping, box
from shapely.ops import unary_union
from rasterio.mask import mask


def extrude_polygons_to_ply(results, ground_heights, output_ply_path, fallback_indices=None, shift_to_local=False):
    meshes = []

    # Optional: global offset
    if shift_to_local:
        all_coords = [pt for res in results for pt in res["polygon"]["coordinates"][0]]
        offset = np.min(np.array(all_coords), axis=0)
    else:
        offset = np.array([0.0, 0.0])

    for i, (entry, ground) in enumerate(zip(results, ground_heights)):
        height = entry["height"]
        extrusion = height - ground
        if extrusion <= 0:
            continue

        polygon = shape(entry["polygon"])
        if shift_to_local:
            shifted = [(x - offset[0], y - offset[1]) for x, y in polygon.exterior.coords]
            polygon = Polygon(shifted)

        try:
            mesh = trimesh.creation.extrude_polygon(polygon, extrusion)
            mesh.apply_translation([0, 0, ground])

            # Set vertex colors (RGBA)
            if fallback_indices and i in fallback_indices:
                color = [255, 0, 0, 255]  # Red
            else:
                color = [200, 200, 200, 255]  # Gray

            vertex_colors = np.tile(color, (len(mesh.vertices), 1))
            mesh.visual.vertex_colors = vertex_colors

            meshes.append(mesh)
        except Exception as e:
            print(f"Failed to extrude polygon {i}: {e}")

    if not meshes:
        print("No valid meshes generated.")
        return

    full = trimesh.util.concatenate(meshes)
    full.export(output_ply_path, file_type="ply", encoding="binary_little_endian")
    print(f"Exported with vertex colors to {output_ply_path}")

def find_matching_tif(polygons, tif_folder, offset=np.array([32000000.0, 0.0])):
    """
    Find TIFFs whose bounding box intersects the bounding box of the adjusted polygons.
    Assumes polygons are in a globally offset system.
    """
    # Apply inverse offset to all polygons
    adjusted_polygons = []
    for poly in polygons:
        shifted = Polygon([(x - offset[0], y - offset[1]) for x, y in poly.exterior.coords])
        adjusted_polygons.append(shifted)

    combined_bounds = unary_union(adjusted_polygons).bounds
    polygon_bbox = box(*combined_bounds)

    matching_tifs = []
    for filename in os.listdir(tif_folder):
        if filename.lower().endswith(".tif"):
            tif_path = os.path.join(tif_folder, filename)
            try:
                with rasterio.open(tif_path) as src:
                    tif_bounds = src.bounds
                    tif_bbox = box(*tif_bounds)
                    if tif_bbox.intersects(polygon_bbox):
                        matching_tifs.append(tif_path)
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

    return matching_tifs


def get_mean_dtm_heights_from_polygons(polygons, tif_list, offset=np.array([32000000.0, 0.0])):
    """
    Returns:
        - list of heights
        - list of indices where fallback (mean of full DTM) was used
    """
    mean_heights = []
    fallback_indices = []

    # Shift polygons back to DTM CRS
    adjusted_polygons = [
        Polygon([(x - offset[0], y - offset[1]) for x, y in poly.exterior.coords])
        for poly in polygons
    ]

    for idx, polygon in enumerate(adjusted_polygons):
        poly_bbox = box(*polygon.bounds)
        matched_dtm = None

        for tif_path in tif_list:
            with rasterio.open(tif_path) as src:
                tif_bbox = box(*src.bounds)
                if tif_bbox.intersects(poly_bbox):
                    matched_dtm = tif_path
                    break

        if not matched_dtm:
            print(f"No DTM match for polygon {idx}, using fallback height.")
            # Fallback: use mean of first available DTM
            with rasterio.open(tif_list[0]) as src:
                data = src.read(1)
                valid = data[data != src.nodata]
                fallback_mean = float(np.mean(valid)) if valid.size > 0 else 0
                mean_heights.append(fallback_mean)
                fallback_indices.append(idx)
            continue

        try:
            with rasterio.open(matched_dtm) as src:
                out_image, _ = mask(src, [mapping(polygon)], crop=True)
                data = out_image[0]
                masked_data = data[data != src.nodata]
                mean_val = float(np.mean(masked_data)) if masked_data.size > 0 else 0
                mean_heights.append(mean_val)
        except Exception as e:
            print(f"Error processing polygon {idx}: {e}")
            mean_heights.append(0)
            fallback_indices.append(idx)

    return mean_heights, fallback_indices

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path= os.path.join(base_dir, "json_outputs")
    dgm_tile="dgm1_32_293_5629_1_nw_2022.tif"
    dgm_path = os.path.join(base_dir, "dgm")
    print(dgm_path)
    with open(os.path.join(json_path,"polygon_data.json") ) as f:
        results = json.load(f)

    polygons = [shape(entry["polygon"]) for entry in results]
    combined_bounds = unary_union(polygons).bounds
    print("Polygon bounds:", combined_bounds)
    matching_tif = find_matching_tif(polygons, dgm_path)
    print(matching_tif)
    ground_heights,missing_height_indices = get_mean_dtm_heights_from_polygons(polygons, matching_tif)
    ground_heights = np.array(ground_heights)
    valid_heights = ground_heights[ground_heights > 0]
    mean_height = valid_heights.mean()
    ground_heights[ground_heights <= 0] = mean_height
    print(ground_heights)
    """ground_heights = [
        np.random.randint(low=190, high=max(200, int(entry["height"])))  # avoid zero height range
        for entry in results
    ]"""

    extrude_polygons_to_ply(results, ground_heights, output_ply_path="test.ply",shift_to_local=True, fallback_indices=missing_height_indices)