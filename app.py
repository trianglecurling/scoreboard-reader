"""Scoreboard reader"""

import os
import cv2
import numpy as np

from PIL import Image
import pytesseract

# There's probably a better way...


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
tl = "template-tl.jpg"
tr = "template-tr.jpg"
bl = "template-bl.jpg"
br = "template-br.jpg"

offset_tl = (16, 11)
offset_tr = (37, 14)
offset_bl = (14, 22)
offset_br = (32, 9)

# Load images
color = cv2.imread(sample_image_path, cv2.IMREAD_COLOR)
orig = color.copy()
# gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
# hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)

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

# Draw circles around the best match points
# cv2.circle(apply_tl, maxloc_tl, 15, 255, 2)
# cv2.circle(apply_tr, maxloc_tr, 15, 255, 2)
# cv2.circle(apply_bl, maxloc_bl, 15, 255, 2)
# cv2.circle(apply_br, maxloc_br, 15, 255, 2)

sbpos_tl = addTuples(maxloc_tl, offset_tl)
sbpos_tr = addTuples(maxloc_tr, offset_tr)
sbpos_bl = addTuples(maxloc_bl, offset_bl)
sbpos_br = addTuples(maxloc_br, offset_br)

# Visualize scoreboard corners with rectangles
# rect_size = 10
# rectpos_tl = addTuples(sbpos_tl, (0, 0))
# rectpos_tr = addTuples(sbpos_tr, (-rect_size, 0))
# rectpos_bl = addTuples(sbpos_bl, (0, -rect_size))
# rectpos_br = addTuples(sbpos_br, (-rect_size, -rect_size))

# cv2.rectangle(color, rectpos_tl, addTuples(rectpos_tl, (rect_size, rect_size)), (255, 0, 0), 2, 8, 0)
# cv2.rectangle(color, rectpos_tr, addTuples(rectpos_tr, (rect_size, rect_size)), (255, 0, 0), 2, 8, 0)
# cv2.rectangle(color, rectpos_bl, addTuples(rectpos_bl, (rect_size, rect_size)), (255, 0, 0), 2, 8, 0)
# cv2.rectangle(color, rectpos_br, addTuples(rectpos_br, (rect_size, rect_size)), (255, 0, 0), 2, 8, 0)

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
cv2.line(visualizer, (0, red_divider_y), (visualizer.shape[1], red_divider_y), (0, 0, 255), 1)
cv2.line(visualizer, (0, yellow_divider_y), (visualizer.shape[1], yellow_divider_y), (0, 255, 255), 1)

# Vertical lines
offset_left = 30
cell_width = 15.85
for i in range(13):
    left = offset_left + round(cell_width * i)
    cv2.line(visualizer, (left, 0), (left, visualizer.shape[0]), 0, 1)

# Extract ROIs
red_cells = []
yellow_cells = []
gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
for i in range(12):
    left_with_extra_room = offset_left + round(cell_width * i) - 4
    right_with_extra_room = left_with_extra_room + round(cell_width) + 8
    red_cells.append(thresh[0:red_divider_y, left_with_extra_room:right_with_extra_room])
    yellow_cells.append(thresh[yellow_divider_y:, left_with_extra_room:right_with_extra_room])

cv2.imshow("color", color)
cv2.imshow("corrected", corrected)

for x, i in enumerate(red_cells):
    print("Red " + str(x + 1) + ": " + pytesseract.image_to_string(i, config="-c tessedit"
                                                                   "_char_whitelist=0123456789 "
                                                                   " --psm 10"
                                                                   " -l osd"
                                                                   " "))
    cv2.imshow("Red " + str(x), i)
for x, i in enumerate(yellow_cells):
    print("Yellow " + str(x + 1) + ": " + pytesseract.image_to_string(i, config="-c tessedit"
                                                                      "_char_whitelist=0123456789 "
                                                                      " --psm 10"
                                                                      " -l osd"
                                                                      " "))
    cv2.imshow("Yellow " + str(x), i)

# cv2.imshow("tlapply", apply_tl)
# cv2.imshow("trapply", apply_tr)
# cv2.imshow("blapply", apply_bl)
# cv2.imshow("brapply", apply_br)

print("color.shape: ", color.shape)
print("template_tl.shape: ", template_tl.shape)
print("template_tr.shape: ", template_tr.shape)
print("template_bl.shape: ", template_bl.shape)
print("template_br.shape: ", template_br.shape)

print("maxloc_tl: ", maxloc_tl)
print("maxloc_tr: ", maxloc_tr)
print("maxloc_bl: ", maxloc_bl)
print("maxloc_br: ", maxloc_br)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Other stuff
# ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
