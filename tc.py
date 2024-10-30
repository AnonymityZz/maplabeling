import os
import cv2
import easyocr
import numpy as np
from skimage.feature import graycomatrix
from skimage.measure import shannon_entropy

def extract_number_from_filename(filename):
    return ''.join(filter(str.isdigit, filename))

def detect_text_area(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img_rgb)

    if not result:
        print(f"No text detected in {os.path.basename(img_path)}")
        return None

    all_points = []

    for (bbox, text, prob) in result:
        points = np.array(bbox, dtype=int)
        all_points.extend(points)

        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

    if all_points:
        all_points = np.array(all_points)
        overall_rect = cv2.minAreaRect(all_points)
        overall_box = cv2.boxPoints(overall_rect)
        overall_box = np.int0(overall_box)
        return overall_rect
    return None

def correct_text_perspective(img, rect):
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    rect_pts = sorted(box, key=lambda x: (x[0], x[1]))
    top_left, bottom_left = sorted(rect_pts[:2], key=lambda x: x[1])
    top_right, bottom_right = sorted(rect_pts[2:], key=lambda x: x[1])

    width = int(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
    height = int(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
    destination_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')

    source_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    M = cv2.getPerspectiveTransform(source_pts, destination_pts)

    return M
def calculate_glcm_and_entropy(img, rect):
    x, y, w, h = cv2.boundingRect(rect)
    img_height, img_width = img.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    if w <= 0 or h <= 0:
        print(f"Warning: Detected invalid ROI size (w={w}, h={h}). Cannot calculate GLCM and entropy.")
        return None, None

    img_height, img_width = img.shape[:2]
    if x < 0 or y < 0 or (x + w) > img_width or (y + h) > img_height:
        print(f"Warning: ROI is out of image bounds. Image size: ({img_width}, {img_height}), "
              f"ROI position: (x={x}, y={y}, w={w}, h={h})")
        return None, None
    roi = img[y:y+h, x:x+w]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if roi.size == 0:
        print("Warning: ROI is empty. Cannot calculate GLCM and entropy.")
        return None, None

    distances = [1]
    # angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(roi_gray, distances=distances, angles=[3*np.pi/4], symmetric=True, normed=True)

    entropy_value = shannon_entropy(glcm)
    return glcm, entropy_value

def process_images(input_folder_a, input_folder_b, output_folder, output_txt):
    files_a = sorted([f for f in os.listdir(input_folder_a) if f.endswith('.png')])
    files_b = sorted([f for f in os.listdir(input_folder_b) if f.endswith('.png')])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_dict_b = {extract_number_from_filename(f): f for f in files_b}

    with open(output_txt, 'w') as f:
        for file_a in files_a:
            img_path_a = os.path.join(input_folder_a, file_a)
            overall_box = detect_text_area(img_path_a)
            if overall_box is not None:
                img_a = cv2.imread(img_path_a)
                m = correct_text_perspective(img_a, overall_box)
                (h, w) = img_a.shape[:2]
                rotated_img = cv2.warpPerspective(img_a, m, (w, h))
                corrected_img_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
                reader = easyocr.Reader(['en'])
                result = reader.readtext(corrected_img_rgb)
                all_points = []
                adjust_image = rotated_img.copy()
                for (bbox, text, prob) in result:
                    points = np.array(bbox, dtype=int)
                    if points.size > 0:
                        all_points.extend(points.flatten().tolist())
                        adjust_image = cv2.drawContours(adjust_image, [points], 0, (0, 255, 0), 2)
                if all_points:
                    overall_rect = cv2.minAreaRect(all_points)
                    overall_box = cv2.boxPoints(overall_rect)
                    overall_box = np.int0(overall_box)
                number_part = extract_number_from_filename(file_a)
                corresponding_file_b = file_dict_b.get(number_part)

                if corresponding_file_b is not None:
                    img_path_b = os.path.join(input_folder_b, corresponding_file_b)
                    img_b = cv2.imread(img_path_b)
                    transformed_b = cv2.warpPerspective(img_b, m, (w, h))
                    glcm, entropy_value = calculate_glcm_and_entropy(transformed_b, overall_box)

                    if entropy_value is not None:
                        f.write(f"{corresponding_file_b}: {entropy_value}\n")
                        print(f"Calculated entropy for {corresponding_file_b}: {entropy_value}")
                    else:
                        print(f"Skipping entropy calculation for {corresponding_file_b} due to empty ROI.")
                else:
                    print(f"No corresponding image found in folder B for {file_a}.")
            else:
                print(f"No text area detected in {file_a}.")

input_folder_a = 'result_outline2'
input_folder_b = 'image'
output_folder = 'comparison/tc/VFL'
output_txt = 'comparison/tc/VFL/entropy_values135.txt'
process_images(input_folder_a, input_folder_b, output_folder, output_txt)