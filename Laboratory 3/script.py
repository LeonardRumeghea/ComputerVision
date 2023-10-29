# Laboratory 3- 18.10.2023 - Lab 3 - skin detection

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

WORKSPACE_FOLDER = './Laboratory 3/'
GROUND_TRUTH = './Laboratory 3/Images/Ground_Truth/'
PRATHEEPAN_DATASET = './Laboratory 3/Images/Pratheepan_Dataset/'
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]

def show_image(img, label='image', save=True, show=True, folder=WORKSPACE_FOLDER):
        
        if not os.path.exists(folder):
            os.makedirs(folder)

        if save:
            cv2.imwrite(f'{folder}/{label}.jpg', img)

        if show:
            cv2.imshow(label,img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

concat = lambda img1, img2, axis: np.concatenate((img1, img2), axis=axis)


# 1. Detect the “skin-pixels” in a color image. Create a new binary image, the same size as the input
# color image, in which the skin pixels are white (255) and all non-skin pixels are black (0).
# Implement all the below described methods.

# 1.1 A color pixel (R,G,B) is classified as “skin” if:
# R > 95 and G > 40 and B > 20 and max{R,G,B} – min{R,G,B} > 15 and |R-G| > 15 and R > G and R > B

def skin_detection_1(pixel):
    b, g, r = pixel
    return r > 95 and g > 40 and b > 20 and (max(r, g, b) - min(r, g, b) > 15) and (max(r, g) - min(r, g) > 15) and r > g and r > b

# 1.2 An (H,S,V) pixel is classified “skin” if: 0 <= H <= 50 and 0.23 <= S <= 0.68 and 0.35 <= V <= 1.0

# (R,G,B) to (H,S,V) conversion:
    # V = max(R,G,B)
    # S = 0 if V == 0 else (V - min(R,G,B)) / V
    # H = 0 if S == 0 else np.arctan((np.sqrt(3) * (G - B)) / (2 * R - G - B))

def skin_detection_2(pixel):
    b, g, r = pixel[0] / 255, pixel[1] / 255, pixel[2] / 255

    v = max(r, g, b)
    delta = v - min(r, g, b)
    s = 0 if v == 0 else delta / v
    
    h = 0 if delta == 0 else \
        60 * (((g - b) / delta) % 6) if v == r else \
        60 * (((b - r) / delta) + 2) if v == g else \
        60 * (((r - g) / delta) + 4)  # if v == b
    
    return 0 <= h <= 50 and 0.23 <= s <= 0.68 and 0.35 <= v <= 1.0


# 1.3 An (Y,Cb,Cr) pixel is classified “skin” if: 80 < Y & 85 < Cb < 135 & 135 < Cr < 180, Y, Cb, Cr in [0, 255]

# (R,G,B) to (Y,Cb,Cr) conversion:
    # Y = 0.299 * R + 0.587 * G + 0.114 * B
    # Cb = -0.169 * R - 0.3313 * G + 0.5 * B + 128
    # Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

def skin_detection_3(pixel):
    b, g, r = pixel
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

    return 80 < y and 85 < cb < 135 and 135 < cr < 180


def skin_detector(img, ground_truth_img, skin_detection_function):
    # result_image = [[([255, 255, 255] if bicolor else pixel) if not skin_detection_function(pixel) else [255, 255, 255] for pixel in row] for row in img]
    result_colorful = []
    confusion_matrix = np.zeros((2, 2))

    # confusion_matrices = 
    for row_idx, row in enumerate(img):
        row_colorful = []
        for column_idx, pixel in enumerate(row):
            if skin_detection_function(pixel):
                row_colorful.append(WHITE)
                calsify_pixel(WHITE, ground_truth_img[row_idx, column_idx], confusion_matrix)
            else:
                row_colorful.append(pixel)
                calsify_pixel(BLACK, ground_truth_img[row_idx, column_idx], confusion_matrix)
        result_colorful.append(row_colorful)

    return np.array(result_colorful, dtype=np.uint8), confusion_matrix


def apply_detections(img, ground_truth_img, skin_detection_functions):
    results_colorful = []
    confusion_matrices = []
    for skin_detection_function in skin_detection_functions:
        result_colorful, confusion_matrix = skin_detector(
                img=img, 
                ground_truth_img=ground_truth_img, 
                skin_detection_function=skin_detection_function
            )
        results_colorful.append(result_colorful)
        confusion_matrices.append(confusion_matrix)

    return results_colorful, confusion_matrices

is_white = lambda pixel: pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255
is_black = lambda pixel: pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0


def calsify_pixel(pixel, ground_truth_pixel, confusion_matrix):
    if is_white(pixel) and is_white(ground_truth_pixel):
        confusion_matrix[0, 0] += 1
    elif is_white(pixel) and is_black(ground_truth_pixel):
        confusion_matrix[0, 1] += 1
    elif is_black(pixel) and is_white(ground_truth_pixel):
        confusion_matrix[1, 0] += 1
    elif is_black(pixel) and is_black(ground_truth_pixel):
        confusion_matrix[1, 1] += 1


def get_accuracy(confusion_matrix):
    return (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)


def save_confusion_matrix(confusion_matrix, label, folder=WORKSPACE_FOLDER):
    plt.imshow(confusion_matrix, cmap='binary')
    plt.title(f'Confusion matrix - {label}')
    plt.xticks([0, 1], ['Skin Pixels', 'Non-skin Pixels'])
    plt.yticks([0, 1], ['Skin Pixels', 'Non-skin Pixels'])
    plt.xlabel('Predicted by the proposed method')
    plt.ylabel('Ground truth')
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='red')

    plt.colorbar()
    # make plot wider
    plt.gcf().set_size_inches(12, 8)

    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(f'{folder}/Confusion matrix - {label}.jpg')
    plt.close()


def detect(img_name, show_results=True):
    skin_detection_functions = [skin_detection_1, skin_detection_2, skin_detection_3]
    skin_detection_labels = ['(R,G,B)', '(H,S,V)', '(Y,Cb,Cr)']

    image = cv2.imread(f'{PRATHEEPAN_DATASET}{img_name}.jpg')
    ground_truth = cv2.imread(f'{GROUND_TRUTH}{img_name}.png')

    results_colorful, confusion_matrices = apply_detections(
            img=image, 
            ground_truth_img=ground_truth,
            skin_detection_functions=skin_detection_functions
        )

    show_image(
            img=concat(
                img1=concat(img1=image, img2=results_colorful[0], axis=1), 
                img2=concat(img1=results_colorful[1], img2=results_colorful[2], axis=1), 
                axis=0
            ), 
            show=show_results,
            label= f'{img_name} - Skin detection - (R,G,B), (H,S,V), (Y,Cb,Cr)', 
            folder=f'{WORKSPACE_FOLDER}Results\{img_name}'
        )

    for idx, matrix in enumerate(confusion_matrices):
        save_confusion_matrix(
                confusion_matrix=matrix, 
                label=f'{img_name} - {skin_detection_labels[idx]}',
                folder=f'{WORKSPACE_FOLDER}Results\{img_name}'
            )

def main():

    for img in os.listdir(PRATHEEPAN_DATASET):
        img_name = img.split('.')[0]
        print(f'> Processing {img_name}...')
        detect(img_name=img_name, show_results=False)


if __name__ == "__main__":
    main()