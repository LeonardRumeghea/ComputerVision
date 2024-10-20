import cv2
import numpy as np
import time

cam = cv2.VideoCapture('Lane Detection - Test Video')

old_left_top = None
old_left_bottom = None

old_right_top = None
old_right_bottom = None

interval_bound = 10 ** 8

while True:
    ret, frame = cam.read()
    if ret is False:
        break
    
    # Exercise 1 - Open the video file “Lane Detection - Test Video” and play it!
    # cv2.imshow('Exercise 1', frame)
    
    # Exercise 2 - Shrink the frame until you could fit around 12 separate windows of that size on your screen.
    (frame_height, frame_width) = frame.shape[:2]
    new_frame_width = frame_width // 4
    new_frame_height = int(frame_height // 3.5)
    frame = cv2.resize(frame, (new_frame_width, new_frame_height))
    cv2.imshow('Exercise 2 - Small', frame)

    original_frame = frame.copy()

    old_right_top = (3/4 * new_frame_width, 0)
    old_right_bottom = (3/4 * new_frame_width, new_frame_height)

    # Exercise 3 - Convert the frame to Grayscale!
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Exercise 3 - Grayscale', frame)

    # Exercise 4 - Select only the road!
    trapezoid_frame = np.zeros((new_frame_height, new_frame_width), dtype=np.uint8)

    upper_left = (int(new_frame_width * 0.46), int(new_frame_height * 0.77)) 
    upper_right = (int(new_frame_width * 0.54), int(new_frame_height * 0.77))
    bottom_left = (0, new_frame_height)
    bottom_right = (new_frame_width, new_frame_height)

    trapezoid_bounds = np.array([upper_right, bottom_right, bottom_left, upper_left], dtype=np.int32)

    cv2.fillConvexPoly(trapezoid_frame, trapezoid_bounds, 255)
    frame = cv2.bitwise_and(frame, trapezoid_frame)
    cv2.imshow('Exercise 4 - Trapezoid', frame)

    # Exercise 5 - Get a top-down view! (sometimes called a birds-eye view)

    trapezoid_bounds = np.float32(trapezoid_bounds)
    frame_bounds = np.float32([
        [new_frame_width, 0],
        [new_frame_width, new_frame_height],
        [0, new_frame_height],
        [0, 0]
    ])

    perspective_transform = cv2.getPerspectiveTransform(trapezoid_bounds, frame_bounds)   
    frame = cv2.warpPerspective(frame, perspective_transform, (new_frame_width, new_frame_height))
    cv2.imshow('Exercise 5 - Top-Down', frame)
    
    # Exercise 6 - Add a bit of blur!

    frame = cv2.blur(frame, ksize=(5, 5))
    cv2.imshow('Exercise 6 - Blur', frame)

    # Exercise 7 - Do edge detection - the Sobel filter!

    solbel_vertical = np.float32([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    solbel_horizontal = np.transpose(solbel_vertical)

    frame = np.float32(frame)
    frame_vertical = cv2.filter2D(frame, -1, solbel_vertical)
    frame_horizontal = cv2.filter2D(frame, -1, solbel_horizontal)

    frame = np.sqrt(np.square(frame_vertical) + np.square(frame_horizontal))
    frame = cv2.convertScaleAbs(frame)
    cv2.imshow('Exercise 7 - Sobel', frame)

    # Exercise 8 - Binarize the frame!

    _, frame = cv2.threshold(frame, 75, 255, cv2.THRESH_BINARY)
    cv2.imshow('Exercise 8 - Binarize', frame)

    # Exercise 9 - Get the coordinates of street markings on each side of the road!

    frame_copy = frame.copy()
    frame_copy[:, :int(new_frame_width * 0.05)] = 0
    frame_copy[:, int(new_frame_width * 0.95):] = 0
    # frame_copy[int(new_frame_height * 0.95):, :] = 0

    white_points = np.argwhere(frame_copy == 255) 

    left_xs = []
    left_ys = []
    right_xs = []
    right_ys = []

    for point in white_points:
        if point[1] < int(new_frame_width * 0.5):
            left_xs.append(point[1])
            left_ys.append(point[0])
        else:
            right_xs.append(point[1])
            right_ys.append(point[0])

    # Exercise 10 - Find the lines that detect the edges of the lane!

    left_polynomial = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
    right_polynomial = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

    left_top_y = 0
    left_top_x = (left_top_y - left_polynomial[0]) / left_polynomial[1]
    left_top_x = left_top_x if -1 * interval_bound <= left_top_x and left_top_x <= interval_bound else old_left_top[0]
    
    left_bottom_y = new_frame_height
    left_bottom_x = (left_bottom_y - left_polynomial[0]) / left_polynomial[1]
    left_bottom_x = left_bottom_x if -1 * interval_bound <= left_bottom_x and left_bottom_x <= interval_bound else old_left_bottom[0]

    right_top_y = 0
    right_top_x = (right_top_y - right_polynomial[0]) / right_polynomial[1]
    right_top_x = right_top_x if -1 * interval_bound <= right_top_x and right_top_x <= interval_bound else old_right_top[0]

    right_bottom_y = new_frame_height
    right_bottom_x = (right_bottom_y - right_polynomial[0]) / right_polynomial[1]
    right_bottom_x = right_bottom_x if -1 * interval_bound <= right_bottom_x and right_bottom_x <= interval_bound else old_right_bottom[0]

    old_left_top = left_top = (int(left_top_x), int(left_top_y))
    old_left_bottom = left_bottom = (int(left_bottom_x), int(left_bottom_y))

    old_right_top = right_top = (int(right_top_x), int(right_top_y))
    old_right_bottom = right_bottom = (int(right_bottom_x), int(right_bottom_y))

    frame = cv2.line(frame, left_top, left_bottom, (200, 0, 0), 5)
    frame = cv2.line(frame, right_top, right_bottom, (100, 0, 0), 5)

    cv2.imshow('Exercise 10 - Lane Lines', frame)

    # Exercise 11 - Create a final visualization!

    perspective_transform = cv2.getPerspectiveTransform(frame_bounds, trapezoid_bounds)

    blank_frame_left = np.zeros((new_frame_height, new_frame_width), dtype=np.uint8)
    blank_frame_left = cv2.line(blank_frame_left, left_top, left_bottom, (255, 0, 0), 3)
    blank_frame_left = cv2.warpPerspective(blank_frame_left, perspective_transform, (new_frame_width, new_frame_height))

    blank_frame_right = np.zeros((new_frame_height, new_frame_width), dtype=np.uint8)
    blank_frame_right = cv2.line(blank_frame_right, right_top, right_bottom, (255, 0, 0), 3)
    blank_frame_right = cv2.warpPerspective(blank_frame_right, perspective_transform, (new_frame_width, new_frame_height))

    left_pixels = np.argwhere(blank_frame_left == 255)
    right_pixels = np.argwhere(blank_frame_right == 255)

    for pixel in left_pixels:
        original_frame[pixel[0]][pixel[1]] = [50, 50, 250]

    for pixel in right_pixels:
        original_frame[pixel[0]][pixel[1]] = [0, 0, 0]

    cv2.imshow('Exercise 11 - Final Visualization', original_frame)

    # time.sleep(0.01)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


