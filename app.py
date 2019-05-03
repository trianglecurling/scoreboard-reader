"""Scoreboard reader"""

import os
import cv2

# Paths
samples_path = "./samples/"
samples_path_a = os.path.join(samples_path, "a")
samples_path_b = os.path.join(samples_path, "b")
samples_path_c = os.path.join(samples_path, "c")
samples_path_d = os.path.join(samples_path, "d")

# This one looks "nice"
sample_image_name = "100246.jpg"
sample_image_path = os.path.join(samples_path_d, sample_image_name)

# Templates
red_template_name = "red-template.jpg"
yellow_template_name = "yellow-template.jpg"

# Load images
color = cv2.imread(sample_image_path, cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
# hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)

# Load templates
red_template = cv2.imread(red_template_name, cv2.IMREAD_COLOR)
yellow_template = cv2.imread(yellow_template_name, cv2.IMREAD_COLOR)

apply_red = cv2.matchTemplate(color, red_template, cv2.TM_CCOEFF_NORMED)
apply_yellow = cv2.matchTemplate(color, yellow_template, cv2.TM_CCOEFF_NORMED)

minval_red, maxval_red, minloc_red, maxloc_red = cv2.minMaxLoc(apply_red)
minval_yellow, maxval_yellow, minloc_yellow, maxloc_yellow = cv2.minMaxLoc(
    apply_yellow)

# Draw circles around the best match points
cv2.circle(apply_red, maxloc_red, 15, 255, 2)
cv2.circle(apply_yellow, maxloc_yellow, 15, 255, 2)

cv2.imshow("color", color)
cv2.imshow("redtemplateapply", apply_red)
cv2.imshow("yellowtemplateapply", apply_yellow)
# cv2.imshow("redtemplate", red_template)
# cv2.imshow("yellowtemplate", yellow_template)

print(color.shape)
print(red_template.shape)
print(yellow_template.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Other stuff
# ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
