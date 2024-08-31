import cv2
import numpy as np

def grayscale(img):
    """Applies the Grayscale transform"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def draw_lines(img, lines, color, thickness):
    """
    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def find_dominant_lines(lines):
    """
    Identify the dominant lines from the list of lines.
    Here, we'll select the longest lines as the dominant lines.
    """
    if lines is None:
        return []

    # Calculate the length of each line and sort them by length
    lengths = [(x1, y1, x2, y2, np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) for line in lines for x1, y1, x2, y2 in line]
    lengths.sort(key=lambda x: x[4], reverse=True)

    # Consider top 2 longest lines as dominant lines
    dominant_lines = lengths[:2]
    dominant_lines = [[(x1, y1, x2, y2)] for x1, y1, x2, y2, length in dominant_lines]

    return dominant_lines

def weighted_img(img, initial_img, α=0.8, β=1.5, λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be an 8-bit 3-channel image (i.e. RGB) `initial_img` should be the image before any processing.
    The result image is computed as follows:
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, 5)
    edges = canny(blur_gray, 50, 150)
    
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (0, imshape[0]//2), (imshape[1], imshape[0]//2), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    
    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_len = 40
    max_line_gap = 20
    
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    
    line_image = np.zeros((*masked_edges.shape, 3), dtype=np.uint8)
    
    if lines is not None:
        draw_lines(line_image, lines, color=[0, 0, 255], thickness=10)  # Draw all lines in blue
        
        dominant_lines = find_dominant_lines(lines)
        draw_lines(line_image, dominant_lines, color=[0, 0, 255], thickness=20)  # Draw dominant lines in red
        draw_lines(line_image, dominant_lines, color=[255, 0, 0], thickness=10)  # Overlap with blue lines

    result = weighted_img(line_image, image)
    
    return result

def process_video_real_time():
    video_path = 'road_video1.mp4'
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter('processed_road_video.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_image(frame)
        out.write(processed_frame)
        
        cv2.imshow('Processed Video', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Execute the function to process the video in real-time
process_video_real_time()
