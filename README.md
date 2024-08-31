# Lane_detection_system
 
Road Lane Line Detection System
This project is designed to process video footage and detect road lane lines in real-time using OpenCV and basic image processing techniques. The system processes video frames, detects edges, identifies regions of interest, and extracts and highlights the most dominant lane lines.

Features
Grayscale Conversion: Converts images to grayscale to simplify processing.
Gaussian Blur: Reduces noise in the image using a Gaussian Blur filter.
Canny Edge Detection: Identifies edges within the image using the Canny edge detection algorithm.
Region of Interest Masking: Applies a mask to the image to focus on the region where lane lines are expected to be.
Hough Transform: Detects lines in the image using the Hough Line Transform.
Dominant Line Detection: Identifies the most dominant lane lines in the image.
Image Overlay: Overlays detected lane lines on the original image.
Real-Time Video Processing: Processes video footage frame by frame to detect lane lines in real-time.

Prerequisites
To run this project, you need to have Python installed along with the following libraries:

OpenCV (cv2)
NumPy (numpy)

How to Use
Place Your Video: Ensure the video file you want to process is named road_video1.mp4 and is located in the same directory as the script.
Run the Script: Execute the Python script to start processing the video in real-time. The processed video will be displayed in a window, and the output will be saved as processed_road_video.mp4.
Control the Video Playback: Press q to stop the video playback and terminate the script.

Code Overview
grayscale(img): Converts the input image to grayscale.
gaussian_blur(img, kernel_size): Applies Gaussian blur to reduce noise.
canny(img, low_threshold, high_threshold): Performs Canny edge detection.
region_of_interest(img, vertices): Masks the image to focus on the region of interest.
hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): Detects lines using the Hough Line Transform.
draw_lines(img, lines, color, thickness): Draws lines on the image.
find_dominant_lines(lines): Identifies the dominant lane lines.
weighted_img(img, initial_img, α=0.8, β=1.5, λ=0.): Overlays detected lane lines on the original image.
process_image(image): Processes a single image to detect and highlight lane lines.
process_video_real_time(): Processes a video frame by frame in real-time and saves the output.

Output
The output is a video file named processed_road_video.mp4, where the detected lane lines are highlighted.
The processed video will also be displayed in real-time in a separate window.
