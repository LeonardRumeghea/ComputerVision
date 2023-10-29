# 1. Install an image processing/computer vision library on your computer (OpenCV)
import cv2
import numpy as np

folder = './Laboratory 1/'

def show_image(img, label='image'):
    cv2.imshow(label,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 2. Open an image (Lena.tif), display its size, plot/write the image.
img = cv2.imread(f'{folder}lena.tif')
print(img.shape) # (512, 512, 3) -> (height, width, channels) -> 512x512 image with 3 channels (RGB)

show_image(img, 'Lena - original image')

# 3. Apply filters that blur/sharpen the image (https://learnopencv.com/image-filtering-using-convolution-in-opencv/). 
# Test these functions with at least 2 values for the parameters (when the function has at least one parameter).
# Plot the results (or save the images).

# Blur the image using a Gaussian filter (cv2.GaussianBlur)
# The Gaussian filter - removes high frequency components (e.g: noise, edges) from the image
# How it works: each pixel is multiplied with the Gaussian kernel and then summed up
blur = cv2.GaussianBlur(img, (15, 15), 0) # (15, 15) is the kernel size, 0 is the standard deviation in X direction
show_image(blur, 'Lena - blurred image')


# Sharpen the image using a Laplacian filter (cv2.Laplacian)
# CV_64F is the data type of the output image (64-bit float) -> negative values are allowed
laplacian = cv2.Laplacian(img, cv2.CV_64F)
show_image(laplacian, 'Lena - sharpened image')

# 4. Apply the following filter:
#   0 -2  0
#  -2  9 -2
#   0 -2  0

# Create a kernel
kernel = np.array([[0, -2, 0], [-2, 9, -2], [0, -2, 0]])
# Apply the kernel
filtered_image = cv2.filter2D(img, -1, kernel) # -1 is the depth of the output image (same as input image) 
show_image(filtered_image, 'Lena - filtered image')

# 5. Rotate an image using different angles, clockwise and counterclockwise. How can an image rotation function be implemented?

# Rotate the image 90 degrees clockwise
rotated_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
show_image(rotated_image, 'Lena - rotated image 90 degrees clockwise')

# 6. Write a function that crops a rectangular part of an image. The parameters of this function are the position of the upper, 
# left pixel in the image, where the cropping starts, the width and the length of the rectangle.

def crop_image(img, x, y, width, height):

    if x < 0 or y < 0 or width < 0 or height < 0:
        raise ValueError('Parameters must be positive')
    
    if x > img.shape[1] or y > img.shape[0] or x + width > img.shape[1] or y + height > img.shape[0]:
        raise ValueError('Parameters exceed image dimensions')

    return img[y:y+height, x:x+width]

try:
    cropped_image = crop_image(img, 128, 128, 256, 256)
    show_image(cropped_image, 'Lena - cropped image')
except ValueError as e:
    print(e)

# 7. Create an emoticon image (emoji) using OpenCV functions. Include this image in the archive that you’ll send at the end of the semester 
# (save it as your_name.jpg).

def get_eye(img, x, y, image_size, color=(0, 0, 0)):
    radius = int(0.1 * image_size)
    pupil_color = (255, 255, 255)
    cv2.circle(img, (x, y), radius, color, -1, 20)
    cv2.circle(img, (x - int(-0.2 * radius), y - int(0.3 * radius)), int(0.35 * radius), pupil_color, -1, 20)
    cv2.circle(img, (x - int(-0.65 * radius), y - int(-0.1 * radius)), int(0.15 * radius), pupil_color, -1, 20)
    cv2.circle(img, (x - int(0.5 * radius), y - int(-0.4 * radius)), int(0.15 * radius), pupil_color, -1, 20)

def get_mouth(img, x, y, image_size, color=(0, 0, 0)):
    radius = int(0.05 * image_size)
    cv2.ellipse(img, (x - radius, y), (radius, radius), 0, 0, 180, color, 12, 20, 0)
    cv2.ellipse(img, (x + radius, y), (radius, radius), 0, 0, 180, color, 12, 20, 0)

def get_blush(img, x, y, image_size, color=(0, 0, 0)):
    cv2.circle(img, (x, y), int(0.04 * image_size), color, -1, 20)

def get_decorative_lines(img, x, y):
    color = (255, 255, 255)
    cv2.ellipse(img, (x, y), (180, 180), 280, 0, 70, color, 12, 20, 0)
    cv2.ellipse(img, (x, y), (180, 180), 270, 0, 1, color, 12, 20, 0)

def get_emoji():
    # Create a black image with 3 channels (RGB) and 512x512 pixels
    image_size = 512
    emoji_radius = int(0.5 * image_size)
    img = np.zeros((image_size, image_size, 3), np.uint8)

    # Define colors used in the emoji
    gray_color = (51, 51, 51)
    face_color = (57, 178, 247)
    blush_color = (40, 84, 249)
    shadow_color = (45, 155, 224)

    # Draw face skin and shadow
    cv2.circle(img, (emoji_radius, emoji_radius), 200, shadow_color, -1, 20) 
    cv2.circle(img, (emoji_radius, emoji_radius), 200, face_color, -1, 20)

    # Create a mask to remove everything outside the face
    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (emoji_radius, emoji_radius), 200, (255, 255, 255), -1)

    # Apply the mask
    img = cv2.bitwise_and(img, mask)

    # Draw the face outline
    cv2.circle(img, (emoji_radius, emoji_radius), 200, gray_color, 10, 20) 

    get_eye(img, int(0.65 * image_size), int(0.45 * image_size), image_size, gray_color)
    get_eye(img, int(0.35 * image_size), int(0.45 * image_size), image_size, gray_color)

    get_mouth(img, int(0.5 * image_size), int(0.6 * image_size), image_size, gray_color)

    get_blush(img, int(0.275 * image_size), int(0.575 * image_size), image_size, blush_color)
    get_blush(img, int(0.725 * image_size), int(0.575 * image_size), image_size, blush_color)

    get_decorative_lines(img, int(0.5 * image_size), int(0.5 * image_size))

    alpha = np.uint8((np.sum(img, axis=-1) > 0) * 255)
    return cv2.merge((img, alpha))

emoji = get_emoji()

# save the image as a jpg file in the current directory with the name 'leonard_rumeghea.jpg'
cv2.imwrite(f'{folder}leonard_rumeghea.png', emoji)
cv2.imwrite(f'{folder}leonard_rumeghea.jpg', emoji)

# open the jpg file with the default image viewer
import os
os.startfile('leonard_rumeghea.png')