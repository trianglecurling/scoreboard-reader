"""Scoreboard reader"""

import os
import cv2
import numpy as np
import pytesseract as tess
from PIL import Image
# tess.pytesseract.tesseract_cmd = r'C:\Tesseract-OCR\tesseract.exe'

def addTuples(a, b):
    if len(a) != len(b):
        raise ValueError("a and b must be the same length")

    result = list(a)
    for i, v in enumerate(b):
        result[i] += v

    return tuple(result)

# From https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

# From https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def getNumber(image):

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY())

    # Otsu Tresholding automatically find best threshold value
    _, binary_image = cv2.threshold('gray', 0, 255, cv2.THRESH_OTSU)

    # invert the image if the text is white and background is black
    count_white = np.sum(binary_image > 0)
    count_black = np.sum(binary_image == 0)
    if count_black > count_white:
        binary_image = 255 - binary_image

    # padding
    final_image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    txt = tess.image_to_string(
        final_image, config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')

    return txt
# Paths
samples_path = "./samples/"
samples_path_a = os.path.join(samples_path, "a")
samples_path_b = os.path.join(samples_path, "b")
samples_path_c = os.path.join(samples_path, "c")
samples_path_d = os.path.join(samples_path, "d")

# This one looks "nice"
sample_image_name = "100119.jpg"  # 100246.jpg is also a good one with fewer cards
sample_image_path = os.path.join(samples_path_d, sample_image_name)

# Templates
tl = "templates/scoreboard-tl.jpg"
tr = "templates/scoreboard-tr.jpg"
bl = "templates/scoreboard-bl.jpg"
br = "templates/scoreboard-br.jpg"

offset_tl = (16, 11)
offset_tr = (37, 14)
offset_bl = (14, 22)
offset_br = (32, 9)

# Load images
color = cv2.imread(sample_image_path, cv2.IMREAD_COLOR)
orig = color.copy()

# Load templates
template_tl = cv2.imread(tl, cv2.IMREAD_COLOR)
template_tr = cv2.imread(tr, cv2.IMREAD_COLOR)
template_bl = cv2.imread(bl, cv2.IMREAD_COLOR)
template_br = cv2.imread(br, cv2.IMREAD_COLOR)

shape_tl = template_tl.shape
shape_tr = template_tr.shape
shape_bl = template_bl.shape
shape_br = template_br.shape

apply_tl = cv2.matchTemplate(color, template_tl, cv2.TM_CCOEFF_NORMED)
apply_tr = cv2.matchTemplate(color, template_tr, cv2.TM_CCOEFF_NORMED)
apply_bl = cv2.matchTemplate(color, template_bl, cv2.TM_CCOEFF_NORMED)
apply_br = cv2.matchTemplate(color, template_br, cv2.TM_CCOEFF_NORMED)

minval_tl, maxval_tl, minloc_tl, maxloc_tl = cv2.minMaxLoc(apply_tl)
minval_tr, maxval_tr, minloc_tr, maxloc_tr = cv2.minMaxLoc(apply_tr)
minval_bl, maxval_bl, minloc_bl, maxloc_bl = cv2.minMaxLoc(apply_bl)
minval_br, maxval_br, minloc_br, maxloc_br = cv2.minMaxLoc(apply_br)

sbpos_tl = addTuples(maxloc_tl, offset_tl)
sbpos_tr = addTuples(maxloc_tr, offset_tr)
sbpos_bl = addTuples(maxloc_bl, offset_bl)
sbpos_br = addTuples(maxloc_br, offset_br)

# Draw 4 lines to outline the scoreboard
cv2.line(color, sbpos_tl, sbpos_tr, (255, 0, 0), 2)
cv2.line(color, sbpos_tr, sbpos_br, (255, 0, 0), 2)
cv2.line(color, sbpos_br, sbpos_bl, (255, 0, 0), 2)
cv2.line(color, sbpos_bl, sbpos_tl, (255, 0, 0), 2)

corrected = four_point_transform(orig, np.array([sbpos_tl, sbpos_tr, sbpos_bl, sbpos_br], dtype="float32"))
visualizer = corrected.copy()

# Horizontal lines
red_divider_y = round(visualizer.shape[0] / 3)
yellow_divider_y = red_divider_y * 2

# Vertical lines
offset_left = 30
cell_width = 15.85

# Extract ROIs
red_cells = []
yellow_cells = []
for i in range(12):
    left_with_extra_room = offset_left + round(cell_width * i) - 0
    right_with_extra_room = left_with_extra_room + round(cell_width) + 4
    red_roi = corrected[0:red_divider_y, left_with_extra_room:right_with_extra_room]
    yellow_roi = corrected[yellow_divider_y:, left_with_extra_room:right_with_extra_room]
    red_cells.append(red_roi)
    yellow_cells.append(yellow_roi)
    
# OCR red_cells and yellow_cells

# OCR red_cells and yellow_cells
text_r = []
conf = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'  
for j in range(12):
    str_temp=tess.image_to_string(Image.fromarray(red_cells[j], 'RGB'),config=conf)
    if str.isdigit(str_temp):
        text_r.append(str_temp)
    else:
        text_r.append('-0')
       
text_y = []      
for j in range(12):
    str_temp=tess.image_to_string(Image.fromarray(yellow_cells[j], 'RGB'),config=conf)
    if str.isdigit(str_temp):
        text_y.append(str_temp)
    else:
        text_y.append('-0')

   
print(text_r)
print(text_y)
 
#Assamptions

flag = True
eightCheck = 0
for j in range(12):
    if (int(text_r[j])<int(text_r[j+1])):
        flag=False
    else:
        print('Invalid because of order')
        break
    if (int(text_y[j])<int(text_y[j+1])):
        flag=False
    else:
        print('Invalid because of order')
        break  
   
    if text_r[j]=='-0':
        eightCheck=eightCheck+1
    else:
        eightCheck=0
   
    if text_y[j]=='-0':
        eightCheck=eightCheck+1
    else:
        eightCheck=0
    if eightCheck>0:
        print('Invalid because of eight consective gaps')
        break



# cv2.imshow("color", color)
# cv2.imshow("corrected", corrected)

# cv2.waitKey(0)
# cv2.destroyAllWindows()