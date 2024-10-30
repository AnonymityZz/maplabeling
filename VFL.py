import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage import measure
from scipy.spatial import distance
import scipy.ndimage as ndi
import matlab.engine
import pandas as pd
import math
import networkx as nx

number = 83
distance_threshold = 25
slope_threshold = 1
text = "The Beauty Studio"
image_path = r"imagedata\data{}.png".format(number)
image = cv2.imread(image_path)
mask_path = r"imagedata\mask{}.png".format(number)
mask = cv2.imread(mask_path)

gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
_, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)
cropped_image = cv2.bitwise_and(image, image, mask=binary_mask)
crop_image_path = r"process\building{}.png".format(number)
cv2.imwrite(crop_image_path, cropped_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

coords = corner_peaks(corner_harris(gray_mask), min_distance=10, threshold_rel=0.3)
coords_subpix = corner_subpix(gray_mask, coords, window_size=15)


coords_list = list(coords_subpix)
nan_rows = [i for i, row in enumerate(coords_list) if any(np.isnan(row))]
coords_list = [row for i, row in enumerate(coords_list) if i not in nan_rows]
contours = measure.find_contours(binary_mask, 0.5)

def flatten_list(lst):
    flattened = [item for sublist in lst for item in sublist]
    return flattened
flattened_polygon = flatten_list(contours)

def find_nearest_point(point, polygon):
    distances = [distance.euclidean(point.ravel(), vertex.ravel()) for vertex in polygon]
    nearest_index = np.argmin(distances)
    return polygon[nearest_index]
nearest_points = []
for point in coords_list:
    nearest_point = find_nearest_point(point, flattened_polygon)
    nearest_points.append(nearest_point)
nearest_points = np.array(nearest_points)

indices = []
for point in nearest_points:
    closest_index = np.argmin(np.linalg.norm(flattened_polygon - point, axis=1))
    indices.append(closest_index)
indices = np.array(indices)
sorted_nearest_points = nearest_points[np.argsort(indices)]

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point2) - np.array(point1))
lines = []
line_length = []
for i in range(len(sorted_nearest_points)):
    start = sorted_nearest_points[i]
    end = sorted_nearest_points[(i + 1) % len(sorted_nearest_points)]
    length = calculate_distance(start, end)
    line_length.append(length)
    lines.append((start, end))

max_length_index = np.argmax(line_length)
max_length = line_length[max_length_index]
max_length_line = lines[max_length_index]

start_point = max_length_line[0]
end_point = max_length_line[1]
md_slope = (end_point[0] - start_point[0]) / (end_point[1] - start_point[1])

eng = matlab.engine.start_matlab()
result = eng.matlab_kmeans(number)

output_filepath = result[0]
labelMatrix = result[1]
kmeans = cv2.imread(output_filepath)
eng.quit()

kmeans_gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(kmeans_gray, kernel, iterations=10)
skeleton = cv2.ximgproc.thinning(erosion)

skeleton_label = measure.label(skeleton)
skeleton_props = measure.regionprops(skeleton_label)
for prop in skeleton_props:
    pixel_count = prop.area
    print("Region:", prop.area,"length:", pixel_count)
largest_region = max(skeleton_props, key=lambda region: region.area)
lskeleton = np.zeros_like(skeleton)
lskeleton[skeleton_label == largest_region.label] = 255
labelMatrix = np.array(labelMatrix)
labelMatrix = labelMatrix.astype(np.int32)

skeleton_df = pd.DataFrame(lskeleton, columns=None)
skeleton_df.to_csv(r'exceloutput\skeleton{}.csv'.format(number), index=False)
label_df = pd.DataFrame(labelMatrix, columns=None)
label_df.to_csv(r'exceloutput\labelMatrix{}.csv'.format(number), index=False)

labels = []
skeleton_coords = np.transpose(np.nonzero(lskeleton))
for point in skeleton_coords:
    x, y = point
    label = labelMatrix[x, y]
    labels.append(label)
labels = np.unique(labels)

zero_matrix = np.zeros((labelMatrix.shape[0], labelMatrix.shape[1]), dtype=int)
for label in labels:
    indices = np.where(labelMatrix == label)
    zero_matrix[indices] = label
zero_df = pd.DataFrame(zero_matrix, columns=None)
zero_df.to_csv(r'exceloutput\splabel{}.csv'.format(number), index=False)
relabel = measure.label(zero_matrix)
relabel_df = pd.DataFrame(relabel, columns=None)
relabel_df.to_csv(r'exceloutput\relabel{}.csv'.format(number), index=False)

props = measure.regionprops(relabel)
relabel_centroids = []
for prop in props:
    relabel_centroids.append(prop.centroid)

num_centroids = len(relabel_centroids)
adjacency_matrix = np.zeros((num_centroids+1, num_centroids+1), dtype=int)

for i in range(num_centroids):
    centroid = relabel_centroids[i]
    x, y = centroid
    label = relabel[int(x), int(y)]
    for mx in range(relabel.shape[0]):
        for my in range(relabel.shape[1]):
            if relabel[mx, my] == label and (mx, my) != (int(x), int(y)):
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ex, ey = mx + dx, my + dy
                        if 0 <= ex < relabel.shape[0] and 0 <= ey < relabel.shape[1]:
                            neighbor_label = relabel[ex, ey]
                            if neighbor_label != label:
                                adjacency_matrix[i, neighbor_label] = 1
adjacency_matrix1 = adjacency_matrix[:-1, 1:]
admatrix_df1 = pd.DataFrame(adjacency_matrix1,columns=None)
admatrix_df1.to_csv(r'exceloutput\adjacency_matrix{}.csv'.format(number), index=False)

G = nx.Graph()
for i in range(adjacency_matrix1.shape[0]):
    for j in range(adjacency_matrix1.shape[1]):
        if adjacency_matrix1[i, j] == 1:
            centroid1 = relabel_centroids[i]
            centroid2 = relabel_centroids[j]
            distance = math.sqrt((centroid2[0] - centroid1[0])**2 + (centroid2[1] - centroid1[1])**2)
            G.add_edge(i, j, weight=distance)
glabels = {i: str(i+1) for i in range(num_centroids)}

selems = list()

selems.append(np.array([[0, 255, 0], [255, 255, 255], [0, 0, 0]]))
selems.append(np.array([[255, 0, 255], [0, 255, 0], [255, 0, 0]]))
selems.append(np.array([[255, 0, 255], [0, 255, 0], [0, 255, 0]]))
selems.append(np.array([[0, 255, 0], [255, 255, 0], [0, 0, 255]]))
selems.append(np.array([[0, 255, 0], [0, 255, 0], [255, 0, 255]]))
selems.append(np.array([[0, 0, 255], [255, 255, 255], [0, 255, 0]]))
selems = [np.rot90(selems[i], k=j) for i in range(5) for j in range(4)]
selems.append(np.array([[0, 255, 0], [255, 255, 255], [0, 255, 0]]))
selems.append(np.array([[255, 0, 255], [0, 255, 0], [255, 0, 255]]))
selems.append(np.array([[0, 255, 0], [255, 255, 255], [255, 0, 0]]))
selems.append(np.array([[255, 0, 0], [0, 255, 255], [255, 255, 0]]))
selems.append(np.array([[0, 255, 255], [255, 255, 0], [0, 255, 255]]))
selems.append(np.array([[255, 0, 0], [0, 255, 255], [255, 255, 0]]))
selems.append(np.array([[0, 0, 255], [255, 255, 255], [0, 255, 0]]))
selems.append(np.array([[255, 0, 255], [0, 255, 255], [0, 255, 0]]))
selems.append(np.array([[0, 255, 255], [255, 255, 0], [0, 0, 255]]))
selems.append(np.array([[0, 255, 255], [255, 255, 0], [0, 255, 0]]))
selems.append(np.array([[255, 255, 0], [0, 255, 255], [255, 255, 0]]))
selems.append(np.array([[255, 255, 0], [0, 255, 255], [255, 0, 0]]))
selems.append(np.array([[255, 0, 255], [255, 255, 255], [0, 255, 0]]))
selems.append(np.array([[0, 255, 0], [255, 255, 0], [255, 0, 255]]))
selems.append(np.array([[0, 255, 0], [0, 255, 255], [255, 255, 0]]))
selems.append(np.array([[0, 0, 0], [0, 255, 0], [0, 0, 255]]))
selems.append(np.array([[0, 0, 0], [0, 255, 0], [0, 255, 0]]))
selems.append(np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0]]))
selems.append(np.array([[0, 0, 0], [0, 255, 255], [0, 0, 0]]))
selems.append(np.array([[0, 0, 0], [255, 255, 0], [0, 0, 0]]))
selems.append(np.array([[0, 0, 255], [0, 255, 0], [0, 0, 0]]))
selems.append(np.array([[0, 255, 0], [0, 255, 0], [0, 0, 0]]))
selems.append(np.array([[255, 0, 0], [0, 255, 0], [0, 0, 0]]))


branches = np.zeros_like(lskeleton, dtype=bool)
for selem in selems:
    branches |= ndi.binary_hit_or_miss(lskeleton, selem)

branch_points_x = []
branch_points_y = []
for y in range(branches.shape[0]):
    for x in range(branches.shape[1]):
        if branches[y, x]:
            branch_points_x.append(x)
            branch_points_y.append(y+1)
points_list = list(zip(branch_points_x, branch_points_y))

point_node_mapping = {}
def get_valid_label(point, relabel):
    x, y = point
    if relabel[y, x] != 0:
        return relabel[y, x]
    neighbors = [
        (x-1, y-1), (x-1, y), (x-1, y+1),
        (x, y-1),           (x, y+1),
        (x+1, y-1), (x+1, y), (x+1, y+1)
    ]
    for nx, ny in neighbors:
        if 0 <= nx < relabel.shape[0] and 0 <= ny < relabel.shape[1]:
            if relabel[ny, nx] != 0:
                return relabel[ny, nx]
    return None
for point in points_list:
    x, y = point
    plabel = get_valid_label((x, y), relabel)
    if plabel is not None:
        centroid = relabel_centroids[plabel-1]
        node = relabel_centroids.index(centroid)
        point_node_mapping[point] = (centroid, node)

edge_distance_list = []
linear_distances_list = []
distance_difference = []
linear_scope_list = []
candidate_lines = []

for i in range(len(points_list)):
    for j in range(i + 1, len(points_list)):
        point1 = points_list[i]
        point2 = points_list[j]
        centroid1, node1 = point_node_mapping[point1]
        centroid2, node2 = point_node_mapping[point2]
        shortest_path = nx.shortest_path(G, source=node1, target=node2)
        gdistance = 0
        for k in range(len(shortest_path) - 1):
            gdistance += G[shortest_path[k]][shortest_path[k + 1]]['weight']
        edge_distance_list.append((node1, node2, gdistance))
        x1, y1 = centroid1
        x2, y2 = centroid2
        ldistance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        linear_distances_list.append((node1, node2, ldistance))
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
        diff_slope = abs(md_slope - slope) / math.sqrt(md_slope ** 2 + slope ** 2)
        linear_scope_list.append((node1+1, node2+1, slope, diff_slope))
        diff = abs(ldistance - gdistance)
        distance_difference.append((node1, node2, diff))
        if diff <distance_threshold and diff_slope <= slope_threshold:
            candidate_lines.append((point1, point2, node1+1, node2+1, ldistance))

output_str1 =""
for line in candidate_lines:
    intersection_pixels = []
    x1, y1 = line[0]
    x2, y2 = line[1]
    node = line[2:4]
    linedistance = line[4]
    points = cv2.line(np.zeros_like(kmeans), (x1, y1), (x2, y2), 255, 1)
    intersection_pixels = np.where((points == 255) & (kmeans == 255))
    num_intersection_pixels = len(intersection_pixels[0])

candidate_lines = [line for line in candidate_lines if line[4] >= 100]
max_length = 0
lettering_line = []

for line in candidate_lines:
    line_length = line[4]

    if line_length > max_length:
        max_length = line_length
        lettering_line = line[:2]

fig, ax = plt.subplots()
cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
ax.imshow(cropped_image)
# instructions to reproduce red lines in Figure 8
ax.plot([max_length_line[0][1], max_length_line[1][1]], [max_length_line[0][0], max_length_line[1][0]], 'r-')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0,0)
plt.savefig(r'direction\direction{}.png'.format(number),dpi=200)
plt.show()

region_mask = cv2.imread(r'process\kmeans_image{}.png'.format(number))
region_mask = cv2.cvtColor(region_mask, cv2.COLOR_BGR2GRAY)
_, binary_region = cv2.threshold(region_mask, 127, 255, cv2.THRESH_BINARY)
region_cropped = cv2.bitwise_and(image, image, mask=binary_region)
crop_region_path = r"process\region{}.png".format(number)
cv2.imwrite(crop_region_path, region_cropped, [cv2.IMWRITE_PNG_COMPRESSION, 0])
background_mask = np.all(region_cropped == [0, 0, 0], axis=2)
gray_image = cv2.cvtColor(region_cropped, cv2.COLOR_BGR2GRAY)
average_brightness = np.mean(gray_image[~background_mask])
print(average_brightness)
if average_brightness > 204:
    font_color = (0, 0, 0)
    add_outline = False
elif average_brightness > 153:
    font_color = (0.3, 0.12, 0)
    add_outline = False
elif average_brightness > 102:
    font_color = (0.63, 0.32, 0.18)
    add_outline = True
elif average_brightness > 51:
    font_color = (0.8, 0.52, 0.25)
    add_outline = True
else:
    font_color = (1, 1, 1)
    add_outline = True
print(font_color)
fig, ax = plt.subplots(dpi=200)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
ax.imshow(image, cmap=plt.cm.gray)
plt.rcParams['font.family'] = 'Arial'
line_start = lettering_line[0]
line_end = lettering_line[1]
line_center = ((line_start[0] + line_end[0]) / 2, (line_start[1] + line_end[1]) / 2)
slope = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])
angle_radians = math.atan(slope)
angle_degrees = math.degrees(angle_radians)
text = ax.text(line_center[0], line_center[1], text, ha='center', va='center',
        rotation=360-angle_degrees, fontsize=8, fontweight='normal', color=font_color)
if add_outline:
    text.set_path_effects([path_effects.Stroke(linewidth=1, foreground=(0, 0, 0)), path_effects.Normal()])
else:
    text.set_path_effects([path_effects.Stroke(linewidth=1, foreground=(1, 1, 1)), path_effects.Normal()])
# instructions to reproduce labeling result in Figure 8
ax.axis('off')
height, width, channels = image.shape
fig.set_size_inches(width/100.0/3.0, height/100.0/3.0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0,0)
plt.savefig(r'result_outline2\result{}.png'.format(number),dpi=200)
plt.show()