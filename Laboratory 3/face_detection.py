# Use skin pixel classification to detect the face in a portrait image (find a minimal square that
# frames the human face). Do not use an already implemented face detection function.

# Implement withouth using opencv face detection functions 

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys

WORKSPACE_FOLDER = './Laboratory 3/'
PORTRETS = './Laboratory 3/Images/Portrets/'
WHITE = np.array([255, 255, 255])
BLACK = np.array([0, 0, 0])

sys.setrecursionlimit(10**6)

def show_image(img):
    cv2.imshow('Face detection' ,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


concat = lambda img1, img2, axis: np.concatenate((img1, img2), axis=axis)


def is_skin(pixel):
    b, g, r = pixel
    return r > 95 and g > 40 and b > 20 and (max(r, g, b) - min(r, g, b) > 15) and (max(r, g) - min(r, g) > 15) and r > g and r > b


def skin_detector(img):
    return np.array([[(WHITE) if is_skin(pixel) else BLACK for pixel in row] for row in img], dtype=np.uint8)

def face_detector_cv(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h),(255, 0, 0), 2)

    return img

def get_bounds(cluster):
    x_min = y_min = 999999
    x_max = y_max = 0

    for x, y in cluster:
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)

    # print(f'x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}')

    return x_min, x_max, y_min, y_max


def dfs(skin_img, visited, x, y, cluster, color_code = 255):
    
    visited[x][y] = 1
    cluster.append((x, y))

    # print(f'x: {x}, y: {y}, len: {len(cluster)}')

    result = not (x - 1 < 0 or x + 1 >= skin_img.shape[0] or y - 1 < 0 or y + 1 >= skin_img.shape[1])

    if x - 1 >= 0 and not visited[x-1][y] and skin_img[x-1][y][0] == color_code:
        dfs(skin_img, visited, x-1, y, cluster, color_code)
    
    if x + 1 < skin_img.shape[0] and not visited[x+1][y] and skin_img[x+1][y][0] == color_code:
        dfs(skin_img, visited, x+1, y, cluster, color_code)

    if y - 1 >= 0 and not visited[x][y-1] and skin_img[x][y-1][0] == color_code:
        dfs(skin_img, visited, x, y-1, cluster, color_code)

    if y + 1 < skin_img.shape[1] and not visited[x][y+1] and skin_img[x][y+1][0] == color_code:
        dfs(skin_img, visited, x, y+1, cluster, color_code)

    return result


def get_cluster_center(cluster):
    x_sum = y_sum = 0

    for x, y in cluster:
        x_sum += x
        y_sum += y

    return (int(x_sum / len(cluster)), int(y_sum / len(cluster)))


# get the centers of the clusters and determine the eye position
def get_eyes_position(img, clusters):
    eyes = []
    clusters_centers = [get_cluster_center(cluster) for cluster in clusters]

    height, width, _ = img.shape

    for idx, (x, y) in enumerate(clusters_centers):
        if (x < 0.5 * height and y < 0.5 * width):
            for x1, y1 in clusters_centers[idx+1:]:
                if (x1 < 0.5 * height and y1 > 0.5 * width):
                    if abs(x - x1) < 0.1 * height and abs(y - y1) < 0.5 * width:
                        eyes.append((x, y)) 
                        eyes.append((x1, y1))
                    
    cv2.line(img, (0, height // 2), (width, height // 2), (0, 0, 255), 2)
    cv2.line(img, (width // 2, 0), (width // 2, height), (0, 0, 255), 2)

    for x, y in clusters_centers:
        cv2.circle(img, (y, x), 3, (0, 255, 0), 2)

    for x, y in eyes:
        cv2.circle(img, (y, x), 3, (0, 0, 255), 2)
    
    return img
                    




def face_detector(img, skin_img):

    visited = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    clusters = []

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if skin_img[x][y][0] == 0 and visited[x][y] == 0:
                cluster = []
                result = dfs(skin_img, visited, x, y, cluster, 0)
                if result:
                    clusters.append(cluster)

    return get_eyes_position(img, clusters)


def main():

    for file in os.listdir(PORTRETS):
        img = cv2.imread(f'{PORTRETS}/{file}')

        if img.shape[0] > 1000 or img.shape[1] > 1000:
            img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))

        skin_img = skin_detector(cv2.GaussianBlur(img, (9, 9), 0))
        result_img = face_detector(img.copy(), skin_img)

        # show_image(concat(img, result_img, 1))
        show_image(concat(skin_img, result_img, 1))

        # result_opencv_img = face_detector_cv(img)

        # show_image(result_img)
        # show_image(result_opencv_img)
        # show_image(concat(result_img, result_opencv_img, 1))

if __name__ == '__main__':
    main()