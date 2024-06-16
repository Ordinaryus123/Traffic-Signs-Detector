import cv2
import numpy as np

def create_stop_sign_template():
    template = np.ones((100, 100, 3), dtype=np.uint8) * 255
    points = np.array([
        [50, 10],
        [90, 25],
        [90, 75],
        [50, 90],
        [10, 75],
        [10, 25]
    ], np.int32)
    cv2.fillPoly(template, [points], (0, 0, 255))
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('stop_sign_template.jpg', template_gray)

def create_caution_sign_template():
    template = np.ones((100, 100, 3), dtype=np.uint8) * 255
    points = np.array([
        [50, 10],
        [90, 50],
        [50, 90],
        [10, 50]
    ], np.int32)
    cv2.fillPoly(template, [points], (0, 255, 255))
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('caution_sign_template.jpg', template_gray)

def create_yield_sign_template():
    template = np.ones((100, 100, 3), dtype=np.uint8) * 255
    points = np.array([
        [50, 10],
        [90, 80],
        [10, 80]
    ], np.int32)
    cv2.fillPoly(template, [points], (0, 0, 255))
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('yield_sign_template.jpg', template_gray)

def create_speed_limit_sign_template():
    template = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(template, (20, 20), (80, 80), (0, 0, 0), 2)
    cv2.putText(template, '50', (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('speed_limit_sign_template.jpg', template_gray)

def create_no_entry_sign_template():
    template = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.circle(template, (50, 50), 40, (0, 0, 255), -1)
    cv2.line(template, (20, 50), (80, 50), (255, 255, 255), 10)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('no_entry_sign_template.jpg', template_gray)


if __name__ == "__main__":
    # Create all the templates
    create_stop_sign_template()
    create_caution_sign_template()
    create_yield_sign_template()
    create_speed_limit_sign_template()
    create_no_entry_sign_template()

    print("Templates created successfully.")