import os
import extract_roi
import numpy as np
import cv2
import re

training_path = "./training/"
samples_paths = ["./samples/a", "./samples/b", "./samples/c", "./samples/d"]
samples_paths = ["./samples/d"]
files = []
for samples_path in samples_paths :
    files.extend([os.path.abspath(os.path.join(samples_path, x)) for x in next(os.walk(samples_path))[2]])

training_file_names = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
blank_count = 0
occluded_count = 0

def check_path_and_write_image(path_to_write, image) :
    if int(path_to_write[path_to_write.rfind("/") + 1:path_to_write.rfind(".png")]) > 100 :
        return
    dir_to_write = os.path.dirname(path_to_write)
    if not os.path.exists(dir_to_write) :
        os.mkdir(dir_to_write)
    cv2.imwrite(path_to_write, image)

def is_blank_cell(image) :
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    meanval = np.mean(thresh)
    # print("meanval: " + str(meanval))
    return meanval > 240


def get_manual_label(image) :
    global training_file_names
    global blank_count
    global occluded_count

    if is_blank_cell(image) :
        training_image_file_name = str(blank_count) + ".png"
        blank_count += 1
        path_to_write = os.path.join(training_path, "blank", training_image_file_name)
    else :
        cv2.imshow("trial", image)
        result = cv2.waitKey(0)
        cv2.destroyWindow("trial")
        num_entered = result - 48
        if 0 <= num_entered <= 9 :
            training_image_file_name = str(training_file_names[num_entered]) + ".png"
            training_file_names[num_entered] += 1
            path_to_write = os.path.join(training_path, str(num_entered), training_image_file_name)
        elif result == 32 :
            training_image_file_name = str(blank_count) + ".png"
            blank_count += 1
            path_to_write = os.path.join(training_path, "blank", training_image_file_name)
        elif result == 120 :
            training_image_file_name = str(occluded_count) + ".png"
            occluded_count += 1
            path_to_write = os.path.join(training_path, "occluded", training_image_file_name)
    check_path_and_write_image(path_to_write, image)


# problem children:
# a/227 226 233 235

for i, filename in enumerate(files) :
    print("file %s of %s (%s%%) - %s" % (i, len(files), round(i / len(files), 2) * 100, filename))
    
    rois = extract_roi.extract_rois(filename)
    if "error" in rois :
        print(rois["error"])
        continue

    red = rois["red"]
    yellow = rois["yellow"]
    full = rois["full"]
    
    # cv2.imshow(filename, full)
    # full_key_result = cv2.waitKey(0)
    # cv2.destroyWindow(filename)

    # if (full_key_result == 27) :
    #     break

    # if (full_key_result == 32) :
    #     # all are blank, just skip (we will see plenty of blanks elsewhere)
    #     continue

    for i in range(12) :
        ret = get_manual_label(red[i])
        # backwards and forwards
    for i in range(12) :
        ret = get_manual_label(yellow[i])
