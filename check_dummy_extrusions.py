import trimesh
from shapely.geometry import shape, Polygon
import numpy as np
import os
import json



def extrude_polygons_to_obj(results, ground_heights, output_obj_path, shift_to_local=False):
    meshes = []

    for entry, ground in zip(results, ground_heights):
        height = entry["height"]
        extrusion = height - ground
        if extrusion <= 0:
            continue  # skip degenerate cases

        polygon = shape(entry["polygon"])  # convert GeoJSON to shapely polygon
        if shift_to_local==True:
            all_coords = [pt for res in results for pt in res["polygon"]["coordinates"][0]]
            all_coords_np = np.array(all_coords)
            offset = all_coords_np.min(axis=0)

            # 2. Shift all polygons using the same offset


            # Shift all coordinates
            def shift_coords(coords, offset):
                return [(x - offset[0], y - offset[1]) for x, y in coords]

            shifted_exterior = shift_coords(polygon.exterior.coords, offset)
            polygon = Polygon(shifted_exterior)

        try:
            mesh = trimesh.creation.extrude_polygon(polygon, extrusion)
            mesh.apply_translation([0, 0, ground])
            meshes.append(mesh)
        except Exception as e:
            print(f"Failed to extrude one polygon: {e}")

    if not meshes:
        print("No valid meshes generated.")
        return

    scene = trimesh.util.concatenate(meshes)
    scene.export(output_obj_path)
    print(f"Exported to {output_obj_path}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path= os.path.join(base_dir, "json_outputs")

    with open(os.path.join(json_path,"polygon_data.json") ) as f:
        results = json.load(f)

    ground_heights = [
        np.random.randint(low=190, high=max(200, int(entry["height"])))  # avoid zero height range
        for entry in results
    ]

    extrude_polygons_to_obj(results, ground_heights, output_obj_path="test.obj",shift_to_local=True)