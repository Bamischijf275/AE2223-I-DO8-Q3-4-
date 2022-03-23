import numpy as np
import geojson as GeoJSON
import shapely as shapely

# SIZE OF IMAGE
x_range = 120
y_range = 155


with open("Image cropping/Tape_B_1_1.geojson") as f:
    allobjects = GeoJSON.load(f)

grid = np.zeros([x_range, y_range])



for i in range(len(allobjects.features)):
    for j in range(len(allobjects.features[i].geometry.coordinates[0])):
        x = allobjects.features[i].geometry.coordinates[0][j][0]
        y = allobjects.features[i].geometry.coordinates[0][j][1]
        grid[int(y)][int(x)] = i + 1


print(grid)


