import os
import cv2
import numpy as np
import fpt

templates_root = "./templates/"
template_tl_path = os.path.join(templates_root, "scoreboard-tl.jpg")
template_tr_path = os.path.join(templates_root, "scoreboard-tr.jpg")
template_bl_path = os.path.join(templates_root, "scoreboard-bl.jpg")
template_br_path = os.path.join(templates_root, "scoreboard-br.jpg")

# Load templates
template_tl = cv2.imread(template_tl_path, cv2.IMREAD_COLOR)
template_tr = cv2.imread(template_tr_path, cv2.IMREAD_COLOR)
template_bl = cv2.imread(template_bl_path, cv2.IMREAD_COLOR)
template_br = cv2.imread(template_br_path, cv2.IMREAD_COLOR)

def addTuples(a, b):
    if len(a) != len(b):
        raise ValueError("a and b must be the same length")

    result = list(a)
    for i, v in enumerate(b):
        result[i] += v

    return tuple(result)

def extract_rois(image_path) :
    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    apply_tl = cv2.matchTemplate(color_image, template_tl, cv2.TM_CCOEFF_NORMED)
    apply_tr = cv2.matchTemplate(color_image, template_tr, cv2.TM_CCOEFF_NORMED)
    apply_bl = cv2.matchTemplate(color_image, template_bl, cv2.TM_CCOEFF_NORMED)
    apply_br = cv2.matchTemplate(color_image, template_br, cv2.TM_CCOEFF_NORMED)

    minval_tl, maxval_tl, minloc_tl, maxloc_tl = cv2.minMaxLoc(apply_tl)
    minval_tr, maxval_tr, minloc_tr, maxloc_tr = cv2.minMaxLoc(apply_tr)
    minval_bl, maxval_bl, minloc_bl, maxloc_bl = cv2.minMaxLoc(apply_bl)
    minval_br, maxval_br, minloc_br, maxloc_br = cv2.minMaxLoc(apply_br)

    if maxval_tl < .92 or maxval_tr < .92 or maxval_bl < .92 or maxval_br < .92 :
        return {"error": "Couldn't identify scoreboard for %s" % image_path}

    # offset from the size of the template image
    offset_tl = (16, 11)
    offset_tr = (37, 14)
    offset_bl = (14, 22)
    offset_br = (32, 9)

    sbpos_tl = addTuples(maxloc_tl, offset_tl)
    sbpos_tr = addTuples(maxloc_tr, offset_tr)
    sbpos_bl = addTuples(maxloc_bl, offset_bl)
    sbpos_br = addTuples(maxloc_br, offset_br)

    corrected = fpt.four_point_transform(color_image, np.array([sbpos_tl, sbpos_tr, sbpos_bl, sbpos_br], dtype="float32"))

    red_divider_y = round(corrected.shape[0] / 3)
    yellow_divider_y = red_divider_y * 2
    offset_left = 30
    cell_width = 15.85

    # Extract ROIs
    red_cells = []
    yellow_cells = []
    for i in range(12):
        left_with_extra_room = offset_left + round(cell_width * i)
        right_with_extra_room = left_with_extra_room + round(cell_width) + 4
        red_roi = corrected[0:red_divider_y, left_with_extra_room:right_with_extra_room]
        yellow_roi = corrected[yellow_divider_y:, left_with_extra_room:right_with_extra_room]

        red_cells.append(red_roi)
        yellow_cells.append(yellow_roi)

    return {"red": red_cells, "yellow": yellow_cells, "full": corrected}