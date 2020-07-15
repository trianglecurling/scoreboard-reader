import os
import cv2
import numpy as np
import fpt

templates_root = "./templates/"
template_a_tl_path = os.path.join(templates_root, "scoreboard-a-tl.png")
template_a_tr_path = os.path.join(templates_root, "scoreboard-a-tr.png")
template_a_bl_path = os.path.join(templates_root, "scoreboard-a-bl.png")
template_a_br_path = os.path.join(templates_root, "scoreboard-a-br.png")
template_b_tl_path = os.path.join(templates_root, "scoreboard-b-tl.png")
template_b_tr_path = os.path.join(templates_root, "scoreboard-b-tr.png")
template_b_bl_path = os.path.join(templates_root, "scoreboard-b-bl.png")
template_b_br_path = os.path.join(templates_root, "scoreboard-b-br.png")
template_c_tl_path = os.path.join(templates_root, "scoreboard-c-tl.png")
template_c_tr_path = os.path.join(templates_root, "scoreboard-c-tr.png")
template_c_bl_path = os.path.join(templates_root, "scoreboard-c-bl.png")
template_c_br_path = os.path.join(templates_root, "scoreboard-c-br.png")
template_d_tl_path = os.path.join(templates_root, "scoreboard-d-tl.png")
template_d_tr_path = os.path.join(templates_root, "scoreboard-d-tr.png")
template_d_bl_path = os.path.join(templates_root, "scoreboard-d-bl.png")
template_d_br_path = os.path.join(templates_root, "scoreboard-d-br.png")

# Load templates for scoreboard corners
template_a_tl = cv2.imread(template_a_tl_path, cv2.IMREAD_COLOR)
template_a_tr = cv2.imread(template_a_tr_path, cv2.IMREAD_COLOR)
template_a_bl = cv2.imread(template_a_bl_path, cv2.IMREAD_COLOR)
template_a_br = cv2.imread(template_a_br_path, cv2.IMREAD_COLOR)
template_b_tl = cv2.imread(template_b_tl_path, cv2.IMREAD_COLOR)
template_b_tr = cv2.imread(template_b_tr_path, cv2.IMREAD_COLOR)
template_b_bl = cv2.imread(template_b_bl_path, cv2.IMREAD_COLOR)
template_b_br = cv2.imread(template_b_br_path, cv2.IMREAD_COLOR)
template_c_tl = cv2.imread(template_c_tl_path, cv2.IMREAD_COLOR)
template_c_tr = cv2.imread(template_c_tr_path, cv2.IMREAD_COLOR)
template_c_bl = cv2.imread(template_c_bl_path, cv2.IMREAD_COLOR)
template_c_br = cv2.imread(template_c_br_path, cv2.IMREAD_COLOR)
template_d_tl = cv2.imread(template_d_tl_path, cv2.IMREAD_COLOR)
template_d_tr = cv2.imread(template_d_tr_path, cv2.IMREAD_COLOR)
template_d_bl = cv2.imread(template_d_bl_path, cv2.IMREAD_COLOR)
template_d_br = cv2.imread(template_d_br_path, cv2.IMREAD_COLOR)

# offset from the size of the template image
offset_a_tl = (8, 8)
offset_a_tr = (21, 8)
offset_a_bl = (8, 22)
offset_a_br = (21, 21)
offset_b_tl = (9, 8)
offset_b_tr = (21, 8)
offset_b_bl = (9, 21)
offset_b_br = (21, 21)
offset_c_tl = (9, 8)
offset_c_tr = (21, 8)
offset_c_bl = (9, 21)
offset_c_br = (21, 21)
offset_d_tl = (7, 8)
offset_d_tr = (21, 8)
offset_d_bl = (8, 21)
offset_d_br = (21, 21)

scoreboard_templates = {"a": {"tl": template_a_tl, "tr": template_a_tr, "bl": template_a_bl, "br": template_a_br}, "b": {"tl": template_b_tl, "tr": template_b_tr, "bl": template_b_bl, "br": template_b_br}, "c": {"tl": template_c_tl, "tr": template_c_tr, "bl": template_c_bl, "br": template_c_br}, "d": {"tl": template_d_tl, "tr": template_d_tr, "bl": template_d_bl, "br": template_d_br}}
scoreboard_template_offsets = {"a": {"tl": offset_a_tl, "tr": offset_a_tr, "bl": offset_a_bl, "br": offset_a_br}, "b": {"tl": offset_b_tl, "tr": offset_b_tr, "bl": offset_b_bl, "br": offset_b_br}, "c": {"tl": offset_c_tl, "tr": offset_c_tr, "bl": offset_c_bl, "br": offset_c_br}, "d": {"tl": offset_d_tl, "tr": offset_d_tr, "bl": offset_d_bl, "br": offset_d_br}}

def addTuples(a, b):
    if len(a) != len(b):
        raise ValueError("a and b must be the same length")

    result = list(a)
    for i, v in enumerate(b):
        result[i] += v

    return tuple(result)

def extract_rois(image_path, scoreboard_id) :
    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    apply_tl = cv2.matchTemplate(color_image, scoreboard_templates[scoreboard_id]["tl"], cv2.TM_CCOEFF_NORMED)
    apply_tr = cv2.matchTemplate(color_image, scoreboard_templates[scoreboard_id]["tr"], cv2.TM_CCOEFF_NORMED)
    apply_bl = cv2.matchTemplate(color_image, scoreboard_templates[scoreboard_id]["bl"], cv2.TM_CCOEFF_NORMED)
    apply_br = cv2.matchTemplate(color_image, scoreboard_templates[scoreboard_id]["br"], cv2.TM_CCOEFF_NORMED)

    minval_tl, maxval_tl, minloc_tl, maxloc_tl = cv2.minMaxLoc(apply_tl)
    minval_tr, maxval_tr, minloc_tr, maxloc_tr = cv2.minMaxLoc(apply_tr)
    minval_bl, maxval_bl, minloc_bl, maxloc_bl = cv2.minMaxLoc(apply_bl)
    minval_br, maxval_br, minloc_br, maxloc_br = cv2.minMaxLoc(apply_br)

    if maxval_tl < .92 or maxval_tr < .92 or maxval_bl < .92 or maxval_br < .92 :
        return {"error": "Couldn't identify scoreboard for %s" % image_path}

    sbpos_tl = addTuples(maxloc_tl, scoreboard_template_offsets[scoreboard_id]["tl"])
    sbpos_tr = addTuples(maxloc_tr, scoreboard_template_offsets[scoreboard_id]["tr"])
    sbpos_bl = addTuples(maxloc_bl, scoreboard_template_offsets[scoreboard_id]["bl"])
    sbpos_br = addTuples(maxloc_br, scoreboard_template_offsets[scoreboard_id]["br"])

    cv2.line(color_image, sbpos_tl, sbpos_tr, (255, 0, 0), 1)
    cv2.line(color_image, sbpos_tr, sbpos_br, (255, 0, 0), 1)
    cv2.line(color_image, sbpos_br, sbpos_bl, (255, 0, 0), 1)
    cv2.line(color_image, sbpos_bl, sbpos_tl, (255, 0, 0), 1)

    cv2.imshow("scoreboard_vis", color_image)
    cv2.waitKey(0)
    cv2.destroyWindow("scoreboard_vis")

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