import numpy as np
import geojson as GeoJSON
import shapely as shapely

with open("Tape A 2.geojson") as f:
    allobjects = GeoJSON.load(f)

test = allobjects

