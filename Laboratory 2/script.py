# Laboratory 2 - 11.10.2023 - RGB to Grayscale Conversions

import cv2	
import numpy as np


folder = './Laboratory 2/'
image_width = 900
image_height = 504
original_image = cv2.resize(cv2.imread(f'{folder}image.jpg'), (image_width, image_height))

def show_image(img, label='image', save=True):

    print(f'\n\n\t•Image label: {label} -> pixels from (300, 300) to (305, 305):', end=' ')
    for i in range(300, 305):
        for j in range(300, 305):
            print(img[i][j], end=' ')

    if save:
        cv2.imwrite(f'{folder}{label}.jpg', img)
    cv2.imshow(label,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

show_image(original_image, 'Original image')

# ------------------- 2. Simple averaging -------------------


# 1. Simple averaging -> (R + G + B) / 3
simple_average = np.uint8(np.sum(original_image, axis=2) / 3)
show_image(simple_average, 'Simple average')

# ------------------- 2. Weighted average -------------------

# 2.1. Gray = 0.3 * R + 0.59 * G + 0.11 * B
result_image = np.average(original_image, axis=2, weights=[0.3, 0.59, 0.11])
result_image = np.uint8(result_image)

show_image(result_image, 'Weighted average - 2.1')

# 2.2. Gray = 0.2126 * R + 0.7152 * G + 0.0722 * B
result_image = np.average(original_image, axis=2, weights=[0.2126, 0.7152, 0.0722])
result_image = np.uint8(result_image)

show_image(result_image, 'Weighted average - 2.2')

# 2.3. Gray = 0.299 * R + 0.587 * G + 0.114 * B
result_image = np.average(original_image, axis=2, weights=[0.299, 0.587, 0.114])
result_image = np.uint8(result_image)

show_image(result_image, 'Weighted average - 2.3')

# ------------------- 3. Decomposition -------------------

# Gray = (min(R, G, B) + max(R, G, B)) / 2
result_image = np.uint8((np.min(original_image, axis=2) + np.max(original_image, axis=2)) / 2)
show_image(result_image, 'Desaturation')

# ------------------- 4. Decomposition -------------------

# 4.1 Gray = max(R, G, B)
result_image = np.max(original_image, axis=2)
show_image(result_image, 'Decomposition - 4.1')

# 4.2 Gray = min(R, G, B)
result_image = np.min(original_image, axis=2)
show_image(result_image, 'Decomposition - 4.2')

# ------------------- 5. Single colour channel -------------------

# 5.1 Gray = R
result_image = original_image[:, :, 0]
show_image(result_image, 'Single colour channel - 5.1')

# 5.2 Gray = G
result_image = original_image[:, :, 1]
show_image(result_image, 'Single colour channel - 5.2')

# 5.3 Gray = B
result_image = original_image[:, :, 2]
show_image(result_image, 'Single colour channel - 5.3')

# ------------------- 6. Custom number of grey shades -------------------

for shades in [2, 4, 8, 16, 32, 64, 128, 256]:
    divider = 256 / shades
    result_image = np.uint8(np.round(np.sum(original_image, axis=2) / 3 / divider) * divider)
    show_image(result_image, f'Custom number of grey shades - {shades}', save=False)

# ------------------- 7. Custom number of grey shades with error - Diffusion Dithering -------------------

# 6.1 The Floyd-Steinberg dithering algorithm 
#   #     *    7/16
#  3/16  5/16  1/16

def nearest_color(pixel_color):
    return 0 if pixel_color < 256 / 2 else 256

result_image = np.zeros((image_height, image_width))
gray_image = np.uint8(np.sum(original_image, axis=2) / 3)

for i in range(image_height):
    for j in range(image_width):
        result_image[i][j] = nearest_color(gray_image[i][j])

        error = gray_image[i][j] - result_image[i][j]

        if j + 1 < image_width:
            gray_image[i][j + 1] += error * 7 / 16
        if i + 1 < image_height and j - 1 >= 0:
            gray_image[i + 1][j - 1] += error * 3 / 16
        if i + 1 < image_height:
            gray_image[i + 1][j] += error * 5 / 16
        if i + 1 < image_height and j + 1 < image_width:
            gray_image[i + 1][j + 1] += error * 1 / 16

show_image(result_image, 'Floyd-Steinberg dithering', save=True)


# 6.2 The mask for the Burkes dithering

#   #     *    8/32  4/32
#  2/32  4/32  8/32  4/32

def nearest_color(pixel_color):
    return 0 if pixel_color < 256 / 2 else 256

result_image = np.zeros((image_height, image_width))
gray_image = np.uint8(np.sum(original_image, axis=2) / 3)

for i in range(image_height):
    for j in range(image_width):
        result_image[i][j] = nearest_color(gray_image[i][j])

        error = gray_image[i][j] - result_image[i][j]

        if j + 1 < image_width:
            gray_image[i][j + 1] += error * 8 / 32
        if j + 2 < image_width:
            gray_image[i][j + 2] += error * 4 / 32
        if i + 1 < image_height and j - 2 >= 0:
            gray_image[i + 1][j - 2] += error * 2 / 32
        if i + 1 < image_height and j - 1 >= 0:
            gray_image[i + 1][j - 1] += error * 4 / 32
        if i + 1 < image_height:
            gray_image[i + 1][j] += error * 8 / 32
        if i + 1 < image_height and j + 1 < image_width:
            gray_image[i + 1][j + 1] += error * 4 / 32
        if i + 1 < image_height and j + 2 < image_width:
            gray_image[i + 1][j + 2] += error * 2 / 32

show_image(result_image, 'Burkes dithering', save=True)