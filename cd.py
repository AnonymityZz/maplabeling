import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

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

def calculate_white_area_centroid(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Unable to read image {img_path}")
        return None

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No white area detected in {os.path.basename(img_path)}")
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None

    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])
    return (centroid_x, centroid_y)

def extract_number_from_filename(filename):
    return ''.join(filter(str.isdigit, filename))

def process_images(input_folder_a, input_folder_b):
    files_a = sorted([f for f in os.listdir(input_folder_a) if f.endswith('.png')])
    files_b = sorted([f for f in os.listdir(input_folder_b) if f.endswith('.png')])

    file_dict_b = {extract_number_from_filename(f): f for f in files_b}

    for file_a in files_a:
        number_a = extract_number_from_filename(file_a)
        if number_a in file_dict_b:
            file_b = file_dict_b[number_a]
            img_path_a = os.path.join(input_folder_a, file_a)
            img_path_b = os.path.join(input_folder_b, file_b)

            text_centroid, img_a = calculate_text_centroid(img_path_a)
            white_area_centroid = calculate_white_area_centroid(img_path_b)

            if text_centroid and white_area_centroid:

                cv2.line(img_a, text_centroid, white_area_centroid, (255, 0, 0), 2)
                cv2.circle(img_a, text_centroid, 5, (0, 0, 255), -1)
                cv2.circle(img_a, white_area_centroid, 5, (0, 255, 0), -1)
                output_path = os.path.join(input_folder_a, 'cd_' + file_a)
                cv2.imwrite(output_path, img_a)
                print(f"Processed image saved at: {output_path}")
            else:
                print(f"Skipping image pair: {file_a} and {file_b} due to missing centroids.")
        else:
            print(f"No matching file found for: {file_a}")

input_folder_a = 'maplexs'
input_folder_b = 'labelarea'
process_images(input_folder_a, input_folder_b)