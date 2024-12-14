import glob
import json
import math
import operator
import os
import shutil
import sys

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("Error: pycocotools is not installed. Please install it to proceed.")
    sys.exit(1)

import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

# Coordinate system explanation
"""
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
"""

def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
    Calculate the log-average miss rate (LAMR).

    LAMR is calculated by averaging miss rates at 9 evenly spaced FPPI (False Positives Per Image)
    points between 10e-2 and 10e0, in log-space.

    Args:
        precision (np.array): Precision values.
        fp_cumsum (np.array): Cumulative false positives.
        num_images (int): Total number of images.

    Returns:
        lamr (float): Log-average miss rate.
        mr (np.array): Miss rate.
        fppi (np.array): False positives per image.
    """
    if precision.size == 0:
        return 0, 1, 0

    fppi = fp_cumsum / float(num_images)
    mr = 1 - precision

    # Insert values to handle edge cases
    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Reference FPPI points in log-space
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # Calculate LAMR
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
    return lamr, mr, fppi

def error(msg):
    """
    Print an error message and exit the program.

    Args:
        msg (str): Error message to display.
    """
    print(msg)
    sys.exit(0)

def is_float_between_0_and_1(value):
    """
    Check if a value is a float between 0.0 and 1.0.

    Args:
        value (str or float): Value to check.

    Returns:
        bool: True if the value is a float between 0.0 and 1.0, False otherwise.
    """
    try:
        val = float(value)
        return 0.0 < val < 1.0
    except ValueError:
        return False

def voc_ap(rec, prec):
    """
    Calculate the Average Precision (AP) using the VOC method.

    Args:
        rec (list): Recall values.
        prec (list): Precision values.

    Returns:
        ap (float): Average Precision.
        mrec (list): Monotonically decreasing recall.
        mprec (list): Monotonically decreasing precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mprec = prec[:]

    # Make precision monotonically decreasing
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])

    # Find indices where recall changes
    i_list = [i for i in range(1, len(mrec)) if mrec[i] != mrec[i - 1]]

    # Calculate AP as the area under the curve
    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mprec[i]
    return ap, mrec, mprec

def file_lines_to_list(path):
    """
    Read lines from a file and convert them to a list.

    Args:
        path (str): Path to the file.

    Returns:
        list: List of lines from the file.
    """
    with open(path) as f:
        content = f.readlines()
    return [x.strip() for x in content]

def draw_text_in_image(img, text, pos, color, line_width):
    """
    Draw text on an image.

    Args:
        img (np.array): Image array.
        text (str): Text to draw.
        pos (tuple): Position (x, y) to draw the text.
        color (tuple): Color of the text in BGR format.
        line_width (int): Line width for the text.

    Returns:
        img (np.array): Image with text drawn.
        text_width (int): Width of the drawn text.
    """
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale, color, lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)

def adjust_axes(r, t, fig, axes):
    """
    Adjust the axes of a plot to fit the text.

    Args:
        r: Renderer.
        t: Text object.
        fig: Figure object.
        axes: Axes object.
    """
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    proportion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * proportion])

def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    """
    Draw a horizontal bar plot using Matplotlib.

    Args:
        dictionary (dict): Dictionary of class names and values.
        n_classes (int): Number of classes.
        window_title (str): Title of the window.
        plot_title (str): Title of the plot.
        x_label (str): Label for the x-axis.
        output_path (str): Path to save the plot.
        to_show (bool): Whether to show the plot.
        plot_color (str): Color of the bars.
        true_p_bar (dict): Dictionary of true positives.
    """
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)

    if true_p_bar:
        fp_sorted = [dictionary[key] - true_p_bar[key] for key in sorted_keys]
        tp_sorted = [true_p_bar[key] for key in sorted_keys]
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        plt.legend(loc='lower right')

        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == len(sorted_values) - 1:
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            if i == len(sorted_values) - 1:
                adjust_axes(r, t, fig, axes)

    fig.canvas.set_window_title(window_title)
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)

    init_height = fig.get_figheight()
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)
    height_in = height_pt / dpi
    top_margin = 0.15
    bottom_margin = 0.05
    figure_height = height_in / (1 - top_margin - bottom_margin)
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_label, fontsize='large')
    fig.tight_layout()
    fig.savefig(output_path)
    if to_show:
        plt.show()
    plt.close()

def get_map(MINOVERLAP, draw_plot, score_threshold=0.5, path='./map_out'):
    """
    Calculate mAP (mean Average Precision) for object detection.

    Args:
        MINOVERLAP (float): Minimum overlap threshold for a detection to be considered correct.
        draw_plot (bool): Whether to draw plots.
        score_threshold (float): Confidence score threshold for detections.
        path (str): Path to the evaluation directory.

    Returns:
        mAP (float): Mean Average Precision.
    """
    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    IMG_PATH = os.path.join(path, 'images-optional')
    TEMP_FILES_PATH = os.path.join(path, '.temp_files')
    RESULTS_FILES_PATH = os.path.join(path, 'results')

    show_animation = True
    if os.path.exists(IMG_PATH):
        for dirpath, dirnames, files in os.walk(IMG_PATH):
            if not files:
                show_animation = False
    else:
        show_animation = False

    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)

    if os.path.exists(RESULTS_FILES_PATH):
        shutil.rmtree(RESULTS_FILES_PATH)
    os.makedirs(RESULTS_FILES_PATH)

    if draw_plot:
        try:
            matplotlib.use('TkAgg')
        except:
            pass
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "AP"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "F1"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Recall"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Precision"))
    if show_animation:
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "images", "detections_one_by_one"))

    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if not ground_truth_files_list:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    gt_counter_per_class = {}
    counter_images_per_class = {}

    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error(f"Error: File not found: {temp_path}")
        lines_list = file_lines_to_list(txt_file)
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except ValueError:
                if "difficult" in line:
                    line_split = line.split()
                    _difficult = line_split[-1]
                    bottom = line_split[-2]
                    right = line_split[-3]
                    top = line_split[-4]
                    left = line_split[-5]
                    class_name = " ".join(line_split[:-5])
                    is_difficult = True
                else:
                    line_split = line.split()
                    bottom = line_split[-1]
                    right = line_split[-2]
                    top = line_split[-3]
                    left = line_split[-4]
                    class_name = " ".join(line_split[:-4])

            bbox = f"{left} {top} {right} {bottom}"
            if is_difficult:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        with open(os.path.join(TEMP_FILES_PATH, f"{file_id}_ground_truth.json"), 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = sorted(list(gt_counter_per_class.keys()))
    n_classes = len(gt_classes)

    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            if class_index == 0 and not os.path.exists(temp_path):
                error(f"Error: File not found: {temp_path}")
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    line_split = line.split()
                    bottom = line_split[-1]
                    right = line_split[-2]
                    top = line_split[-3]
                    left = line_split[-4]
                    confidence = line_split[-5]
                    tmp_class_name = " ".join(line_split[:-5])

                if tmp_class_name == class_name:
                    bbox = f"{left} {top} {right} {bottom}"
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})

        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(os.path.join(TEMP_FILES_PATH, f"{class_name}_dr.json"), 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    with open(os.path.join(RESULTS_FILES_PATH, "results.txt"), 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}

        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            dr_file = os.path.join(TEMP_FILES_PATH, f"{class_name}_dr.json")
            dr_data = json.load(open(dr_file))

            nd = len(dr_data)
            tp = [0] * nd
            fp = [0] * nd
            score = [0] * nd
            score_threshold_idx = 0
            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                score[idx] = float(detection["confidence"])
                if score[idx] >= score_threshold:
                    score_threshold_idx = idx

                if show_animation:
                    ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                    if not ground_truth_img:
                        error(f"Error: Image not found with id: {file_id}")
                    elif len(ground_truth_img) > 1:
                        error(f"Error: Multiple images with id: {file_id}")
                    else:
                        img = cv2.imread(os.path.join(IMG_PATH, ground_truth_img[0]))
                        img_cumulative_path = os.path.join(RESULTS_FILES_PATH, "images", ground_truth_img[0])
                        if os.path.isfile(img_cumulative_path):
                            img_cumulative = cv2.imread(img_cumulative_path)
                        else:
                            img_cumulative = img.copy()
                        bottom_border = 60
                        BLACK = [0, 0, 0]
                        img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)

                gt_file = os.path.join(TEMP_FILES_PATH, f"{file_id}_ground_truth.json")
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                bb = [float(x) for x in detection["bbox"].split()]
                for obj in ground_truth_data:
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                if show_animation:
                    status = "NO MATCH FOUND!"

                min_overlap = MINOVERLAP
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                        if not gt_match["used"]:
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                            if show_animation:
                                status = "MATCH!"
                        else:
                            fp[idx] = 1
                            if show_animation:
                                status = "REPEATED MATCH!"
                else:
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

                if show_animation:
                    height, width = img.shape[:2]
                    white = (255, 255, 255)
                    light_blue = (255, 200, 100)
                    green = (0, 255, 0)
                    light_red = (30, 30, 255)
                    margin = 10
                    v_pos = int(height - margin - (bottom_border / 2.0))
                    text = f"Image: {ground_truth_img[0]} "
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    text = f"Class [{class_index}/{n_classes}]: {class_name} "
                    img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
                    if ovmax != -1:
                        color = light_red
                        if status == "INSUFFICIENT OVERLAP":
                            text = f"IoU: {ovmax * 100:.2f}% < {min_overlap * 100:.2f}%"
                        else:
                            text = f"IoU: {ovmax * 100:.2f}% >= {min_overlap * 100:.2f}%"
                            color = green
                        img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                    v_pos += int(bottom_border / 2.0)
                    rank_pos = str(idx + 1)
                    text = f"Detection #rank: {rank_pos} confidence: {float(detection['confidence']) * 100:.2f}%"
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    color = light_red
                    if status == "MATCH!":
                        color = green
                    text = f"Result: {status} "
                    img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if ovmax > 0:
                        bbgt = [int(round(float(x))) for x in gt_match["bbox"].split()]
                        cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                        cv2.rectangle(img_cumulative, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                        cv2.putText(img_cumulative, class_name, (bbgt[0], bbgt[1] - 5), font, 0.6, light_blue, 1, cv2.LINE_AA)
                    bb = [int(i) for i in bb]
                    cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                    cv2.rectangle(img_cumulative, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                    cv2.putText(img_cumulative, class_name, (bb[0], bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)

                    cv2.imshow("Animation", img)
                    cv2.waitKey(20)
                    output_img_path = os.path.join(RESULTS_FILES_PATH, "images", "detections_one_by_one", f"{class_name}_detection{idx}.jpg")
                    cv2.imwrite(output_img_path, img)
                    cv2.imwrite(img_cumulative_path, img_cumulative)

            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val

            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val

            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / max(gt_counter_per_class[class_name], 1)

            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / max((fp[idx] + tp[idx]), 1)

            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            F1 = np.array(rec) * np.array(prec) * 2 / np.where((np.array(prec) + np.array(rec)) == 0, 1, (np.array(prec) + np.array(rec)))

            sum_AP += ap
            text = f"{ap * 100:.2f}% = {class_name} AP"

            if len(prec) > 0:
                F1_text = f"{F1[score_threshold_idx]:.2f} = {class_name} F1"
                Recall_text = f"{rec[score_threshold_idx] * 100:.2f}% = {class_name} Recall"
                Precision_text = f"{prec[score_threshold_idx] * 100:.2f}% = {class_name} Precision"
            else:
                F1_text = f"0.00 = {class_name} F1"
                Recall_text = f"0.00% = {class_name} Recall"
                Precision_text = f"0.00% = {class_name} Precision"

            rounded_prec = [f'{elem:.2f}' for elem in prec]
            rounded_rec = [f'{elem:.2f}' for elem in rec]
            results_file.write(f"{text}\n Precision: {rounded_prec}\n Recall: {rounded_rec}\n\n")

            if len(prec) > 0:
                print(f"{text}\t||\tscore_threshold={score_threshold}: F1={F1[score_threshold_idx]:.2f}; Recall={rec[score_threshold_idx] * 100:.2f}%; Precision={prec[score_threshold_idx] * 100:.2f}%")
            else:
                print(f"{text}\t||\tscore_threshold={score_threshold}: F1=0.00%; Recall=0.00%; Precision=0.00%")
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class[class_name]
            lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
            lamr_dictionary[class_name] = lamr

            if draw_plot:
                plt.plot(rec, prec, '-o')
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')

                fig = plt.gcf()
                fig.canvas.set_window_title(f'AP {class_name}')

                plt.title(f'class: {text}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(os.path.join(RESULTS_FILES_PATH, "AP", f"{class_name}.png"))
                plt.cla()

                plt.plot(score, F1, "-", color='orangered')
                plt.title(f'class: {F1_text}\nscore_threshold={score_threshold}')
                plt.xlabel('Score_Threshold')
                plt.ylabel('F1')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(os.path.join(RESULTS_FILES_PATH, "F1", f"{class_name}.png"))
                plt.cla()

                plt.plot(score, rec, "-H", color='gold')
                plt.title(f'class: {Recall_text}\nscore_threshold={score_threshold}')
                plt.xlabel('Score_Threshold')
                plt.ylabel('Recall')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(os.path.join(RESULTS_FILES_PATH, "Recall", f"{class_name}.png"))
                plt.cla()

                plt.plot(score, prec, "-s", color='palevioletred')
                plt.title(f'class: {Precision_text}\nscore_threshold={score_threshold}')
                plt.xlabel('Score_Threshold')
                plt.ylabel('Precision')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(os.path.join(RESULTS_FILES_PATH, "Precision", f"{class_name}.png"))
                plt.cla()

        if show_animation:
            cv2.destroyAllWindows()
        if n_classes == 0:
            print("未检测到任何种类，请检查标签信息与get_map.py中的classes_path是否修改。")
            return 0
        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes
        text = f"mAP = {mAP * 100:.2f}%"
        results_file.write(text + "\n")
        print(text)

    shutil.rmtree(TEMP_FILES_PATH)

    det_counter_per_class = {}
    for txt_file in dr_files_list:
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            if class_name in det_counter_per_class:
                det_counter_per_class[class_name] += 1
            else:
                det_counter_per_class[class_name] = 1
    dr_classes = list(det_counter_per_class.keys())

    with open(os.path.join(RESULTS_FILES_PATH, "results.txt"), 'a') as results_file:
        results_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(f"{class_name}: {gt_counter_per_class[class_name]}\n")

    for class_name in dr_classes:
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0

    with open(os.path.join(RESULTS_FILES_PATH, "results.txt"), 'a') as results_file:
        results_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = f"{class_name}: {n_det}"
            text += f" (tp:{count_true_positives[class_name]}"
            text += f", fp:{n_det - count_true_positives[class_name]})\n"
            results_file.write(text)

    if draw_plot:
        window_title = "ground-truth-info"
        plot_title = f"ground-truth\n({len(ground_truth_files_list)} files and {n_classes} classes)"
        x_label = "Number of objects per class"
        output_path = os.path.join(RESULTS_FILES_PATH, "ground-truth-info.png")
        to_show = False
        plot_color = 'forestgreen'
        draw_plot_func(
            gt_counter_per_class,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ''
        )

    if draw_plot:
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = os.path.join(RESULTS_FILES_PATH, "lamr.png")
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(
            lamr_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
        )

    if draw_plot:
        window_title = "mAP"
        plot_title = f"mAP = {mAP * 100:.2f}%"
        x_label = "Average Precision"
        output_path = os.path.join(RESULTS_FILES_PATH, "mAP.png")
        to_show = True
        plot_color = 'royalblue'
        draw_plot_func(
            ap_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
        )
    return mAP

def preprocess_gt(gt_path, class_names):
    """
    Preprocess ground truth data and convert it to COCO format.

    Args:
        gt_path (str): Path to the ground truth directory.
        class_names (list): List of class names.

    Returns:
        dict: COCO-formatted ground truth data.
    """
    image_ids = os.listdir(gt_path)
    results = {}

    images = []
    bboxes = []
    for i, image_id in enumerate(image_ids):
        lines_list = file_lines_to_list(os.path.join(gt_path, image_id))
        boxes_per_image = []
        image = {}
        image_id = os.path.splitext(image_id)[0]
        image['file_name'] = image_id + '.jpg'
        image['width'] = 1
        image['height'] = 1
        image['id'] = str(image_id)

        for line in lines_list:
            difficult = 0
            if "difficult" in line:
                line_split = line.split()
                left, top, right, bottom, _difficult = line_split[-5:]
                class_name = " ".join(line_split[:-5])
                difficult = 1
            else:
                line_split = line.split()
                left, top, right, bottom = line_split[-4:]
                class_name = " ".join(line_split[:-4])

            left, top, right, bottom = float(left), float(top), float(right), float(bottom)
            if class_name not in class_names:
                continue
            cls_id = class_names.index(class_name) + 1
            bbox = [left, top, right - left, bottom - top, difficult, str(image_id), cls_id, (right - left) * (bottom - top) - 10.0]
            boxes_per_image.append(bbox)
        images.append(image)
        bboxes.extend(boxes_per_image)
    results['images'] = images

    categories = []
    for i, cls in enumerate(class_names):
        category = {}
        category['supercategory'] = cls
        category['name'] = cls
        category['id'] = i + 1
        categories.append(category)
    results['categories'] = categories

    annotations = []
    for i, box in enumerate(bboxes):
        annotation = {}
        annotation['area'] = box[-1]
        annotation['category_id'] = box[-2]
        annotation['image_id'] = box[-3]
        annotation['iscrowd'] = box[-4]
        annotation['bbox'] = box[:4]
        annotation['id'] = i
        annotations.append(annotation)
    results['annotations'] = annotations
    return results

def preprocess_dr(dr_path, class_names):
    """
    Preprocess detection results and convert them to COCO format.

    Args:
        dr_path (str): Path to the detection results directory.
        class_names (list): List of class names.

    Returns:
        list: COCO-formatted detection results.
    """
    image_ids = os.listdir(dr_path)
    results = []
    for image_id in image_ids:
        lines_list = file_lines_to_list(os.path.join(dr_path, image_id))
        image_id = os.path.splitext(image_id)[0]
        for line in lines_list:
            line_split = line.split()
            confidence, left, top, right, bottom = line_split[-5:]
            class_name = " ".join(line_split[:-5])
            left, top, right, bottom = float(left), float(top), float(right), float(bottom)
            result = {}
            result["image_id"] = str(image_id)
            if class_name not in class_names:
                continue
            result["category_id"] = class_names.index(class_name) + 1
            result["bbox"] = [left, top, right - left, bottom - top]
            result["score"] = float(confidence)
            results.append(result)
    return results

def get_coco_map(class_names, path):
    """
    Calculate COCO evaluation metrics for object detection.

    Args:
        class_names (list): List of class names.
        path (str): Path to the evaluation directory.

    Returns:
        list: COCO evaluation metrics.
    """
    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    COCO_PATH = os.path.join(path, 'coco_eval')

    if not os.path.exists(COCO_PATH):
        os.makedirs(COCO_PATH)

    GT_JSON_PATH = os.path.join(COCO_PATH, 'instances_gt.json')
    DR_JSON_PATH = os.path.join(COCO_PATH, 'instances_dr.json')

    with open(GT_JSON_PATH, "w") as f:
        results_gt = preprocess_gt(GT_PATH, class_names)
        json.dump(results_gt, f, indent=4)

    with open(DR_JSON_PATH, "w") as f:
        results_dr = preprocess_dr(DR_PATH, class_names)
        json.dump(results_dr, f, indent=4)
        if len(results_dr) == 0:
            print("未检测到任何目标。")
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    cocoGt = COCO(GT_JSON_PATH)
    cocoDt = cocoGt.loadRes(DR_JSON_PATH)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats