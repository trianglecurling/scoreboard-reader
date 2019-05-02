"""Scoreboard reader"""

import os
import cv2

print("done")

samples_path = "./samples/"
samples_path_a = os.path.join(samples_path, "a")
samples_path_b = os.path.join(samples_path, "b")
samples_path_c = os.path.join(samples_path, "c")
samples_path_d = os.path.join(samples_path, "d")

# This one looks "nice"
sample_image_name = "100246.jpg"
sample_image_path = os.path.join(samples_path_d, sample_image_name)

sample = cv2.imread(sample_image_path, cv2.IMREAD_COLOR)
cv2.imshow("Sample image", sample)
