from importlib import metadata
import os
import lmdb
import json
import folium
import math

data_path = "/home/hk-project-p0021769/hgf_grc7525/data/ecotaxa/ecotaxa_lmdb/UVP5HD/"
image_path = data_path + "images-labeled"
metadata_path = data_path + "meta-labeled"
label_path = data_path + "labels"

coordinates  = set()

# path = os.path.join(data_path)
env_imgs: lmdb.Environment = lmdb.open(image_path, readonly=True)
env_labels: lmdb.Environment = lmdb.open(label_path, readonly=True)
env_meta: lmdb.Environment = lmdb.open(metadata_path, readonly=True)
counter = 0

with env_imgs.begin() as txn_img , env_meta.begin() as txn_meta, env_labels.begin() as txn_labels:
    cursor_img = txn_img.cursor()

    for key, value_img in cursor_img:
        value_meta: bytes = txn_meta.get(key)
        value_label: bytes = txn_labels.get(key)

        key = key.decode("utf-8")

        value_meta = value_meta.decode("utf-8")
        label = value_label.decode("utf-8") if value_label else None

        meta_json = json.loads(value_meta)

        # lat = meta_json.get("object_lat")
        # lon = meta_json.get("object_lon")

        print(f"Key: {key}, Meta: {meta_json}, label {label}", )
        counter += 1
        if counter > 100:
            break
        # if lat is not None and lon is not None and not math.isnan(lat) and not math.isnan(lon):
        #     coordinates.add((round(lat, 5), round(lon, 5)))
        # print(f"Key: {key}, Value: {value}")

# # print(coordinates)
# print(f"Number of unique coordinates: {len(coordinates)}")
# coordinates = list(coordinates) # Convert set to list for iteration
# print(coordinates)
# avg_lat = sum(lat for lat, lon in coordinates) / len(coordinates)
# avg_lon = sum(lon for lat, lon in coordinates) / len(coordinates)

# # Create a map
# m = folium.Map(location=[avg_lat, avg_lon], zoom_start=5)

# # Add points

# for lat, lon in coordinates:
#     folium.CircleMarker(
#             location=[lat, lon],
#             radius=3,                # size of the dot
#             color='red',            # border color
#             fill=True,
#             fill_color='red',       # fill color
#             fill_opacity=0.7         # transparency
#         ).add_to(m)
# # Save to HTML or display in Jupyter
# m.save("/home/hk-project-p0021769/hgf_col5747/output/map_dots_all.html")