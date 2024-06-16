import cv2
import numpy as np

# Color ranges for HSV
# Red color range for stop sign, yield sign, and no entry sign
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
# Yellow color range for caution sign
lower_yellow = np.array([15, 100, 100])
upper_yellow = np.array([35, 255, 255])

def color_segmentation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Red mask
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    # Yellow mask
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    return mask_red, mask_yellow

def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return edges

def template_matching(image, templates):
    best_match = None
    best_value = 0.0
    for template in templates:
        for scale in np.linspace(0.5, 1.5, 20):
            resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            res = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > best_value:
                best_value = max_val
                best_match = (max_val, max_loc, resized_template.shape)
    return best_match

def shape_analysis(contours, min_vertices=3, max_vertices=10, aspect_ratio_range=(0.8, 1.2)):
    matches = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = len(approx)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if min_vertices <= vertices <= max_vertices and aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
            matches.append(cnt)
    return matches


if __name__ == "__main__":
    #Test image
    image = cv2.imread('stop.jpg')
    # Resize image
    image = cv2.resize(image, (640, 480))
    
    mask_red, mask_yellow = color_segmentation(image)
    edges = sobel_edge_detection(image)
    
    # Find contours from masks
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze shapes
    red_matches = shape_analysis(contours_red)
    yellow_matches = shape_analysis(contours_yellow)