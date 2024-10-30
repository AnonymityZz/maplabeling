import cv2
import easyocr
import numpy as np
import os

def calculate_text_centroid(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img_rgb)

    if not result:
        print(f"No text detected in {os.path.basename(img_path)}")
        return None, img

    min_x, min_y = img.shape[1], img.shape[0]
    max_x, max_y = 0, 0

    for (bbox, text, prob) in result:
        points = np.array(bbox, dtype=int)
        min_x = min(min_x, points[:, 0].min())
        min_y = min(min_y, points[:, 1].min())
        max_x = max(max_x, points[:, 0].max())
        max_y = max(max_y, points[:, 1].max())

    centroid_x = int((min_x + max_x) / 2)
    centroid_y = int((min_y + max_y) / 2)
    return (centroid_x, centroid_y), img

def calculate_longest_skeleton(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Unable to read image {img_path}")
        return None

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=10)
    skeleton = cv2.ximgproc.thinning(erosion)

    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print(f"No skeleton found in {os.path.basename(img_path)}")
        return None

    longest_contour = max(contours, key=cv2.contourArea)

    return longest_contour

def extract_number_from_filename(filename):
    return ''.join(filter(str.isdigit, filename))


def calculate_distance(centroid, contour):
    min_distance = float('inf')
    nearest_point = None

    for point in contour:
        skeleton_point = point[0]
        distance = np.linalg.norm(np.array(centroid) - np.array(skeleton_point))

        if distance < min_distance:
            min_distance = distance
            nearest_point = skeleton_point
    return min_distance, nearest_point

def process_images(input_folder_a, input_folder_b, output_folder):
    files_a = sorted([f for f in os.listdir(input_folder_a) if f.endswith('.png')])
    files_b = sorted([f for f in os.listdir(input_folder_b) if f.endswith('.png')])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_dict_b = {extract_number_from_filename(f): f for f in files_b}

    distance_output_file = os.path.join(output_folder, 'distances.txt')
    with open(distance_output_file, 'w') as distance_file:
        for file_a in files_a:
            number_a = extract_number_from_filename(file_a)
            if number_a in file_dict_b:
                file_b = file_dict_b[number_a]
                img_path_a = os.path.join(input_folder_a, file_a)
                img_path_b = os.path.join(input_folder_b, file_b)

                text_centroid, img_a = calculate_text_centroid(img_path_a)
                longest_skeleton = calculate_longest_skeleton(img_path_b)

                if text_centroid and longest_skeleton is not None:
                    min_distance, nearest_point = calculate_distance(text_centroid, longest_skeleton)
                    cv2.line(img_a, text_centroid, tuple(nearest_point), (255, 0, 0), 2)
                    cv2.drawContours(img_a, [longest_skeleton], -1, (0, 255, 0), 2)
                    cv2.circle(img_a, text_centroid, 5, (0, 0, 255), -1)
                    cv2.circle(img_a, tuple(nearest_point), 5, (0, 255, 255), -1)
                    distance_file.write(f"Distance for {file_a}: {min_distance}\n")

                    output_path = os.path.join(output_folder, 'sd_' + file_a)
                    cv2.imwrite(output_path, img_a)
                    print(f"Processed image saved at: {output_path}")
                else:
                    print(f"Skipping image pair: {file_a} and {file_b} due to missing data.")
            else:
                print(f"No matching file found for: {file_a}")


input_folder_a = 'result_outline2'
input_folder_b = 'labelarea'
output_folder = r'comparison/sd/VFL'
process_images(input_folder_a, input_folder_b, output_folder)