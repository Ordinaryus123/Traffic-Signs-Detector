import cv2
import numpy as np
from image_process import shape_analysis, color_segmentation, sobel_edge_detection, template_matching


# Load templates (predefined)
templates = {
    'stop': cv2.imread('stop_sign_template.jpg', 0),
    'caution': cv2.imread('caution_sign_template.jpg', 0),
    'yield': cv2.imread('yield_sign_template.jpg', 0),
    'no_entry': cv2.imread('no_entry_sign_template.jpg', 0)
}

def detect_traffic_signs(image):
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
    #print(red_matches)
    # Initialize best match variable
    best_match = (0, None, None, None)  # (confidence, (x, y), sign_type, color)
    
    # Template matching for red signs
    for cnt in red_matches:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = edges[y:y+h, x:x+w]
        for sign_type in ['stop', 'yield', 'no_entry']:
            match = template_matching(roi, [templates[sign_type]]) 
            print(str(match[0]) + " "+sign_type) 
            if match and match[0] > best_match[0]: 
                best_match = (match[0], (x, y), sign_type, 'red')
    
    # Template matching for yellow signs
    for cnt in yellow_matches:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = edges[y:y+h, x:x+w]
        match = template_matching(roi, [templates['caution']])
        if match and match[0] > best_match[0]:
            best_match = (match[0], (x, y), 'caution', 'yellow')
    
    return best_match

def annotate_frame(image, best_match):
    if best_match[0] > 0.2:  # Confidence threshold
        confidence, (x, y), sign_type, color = best_match
        cv2.putText(image, f"{sign_type.capitalize()} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(image, (x, y), (x + 100, y + 100), (0, 255, 0), 2)
    return image

if __name__ == "__main__":
    # Test
    image = cv2.imread('yield.jpg') 
    best_match = detect_traffic_signs(image)
    result_image = annotate_frame(image, best_match)
    cv2.imshow('Traffic Sign Detection', result_image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Real-time processing
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            best_match = detect_traffic_signs(frame)
            if(best_match[0] != 0):
                result_frame = annotate_frame(frame, best_match)
            else:
                result_frame = frame
        except:
            result_frame = frame 
        cv2.imshow('Traffic Sign Detection', result_frame) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()