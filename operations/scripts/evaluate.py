import boto3
import json
import logging
import os
import pathlib

import numpy as np

import shared_constants

from typing import List, Tuple, Union, Dict


logging.basicConfig(level=logging.INFO)
s3 = boto3.client('s3')
CONF_THRESHOLD = 0.15


def main():
    calculate_mean_average_precision_score()


def calculate_mean_average_precision_score():
    """
    Calculates the mean average precision (mAP) score for object detection predictions.

    This function reads true labels and bounding boxes from label files in the labels dir,
    and predicted labels and bounding boxes from transform files in transform dir.
    The mAP score is then calculated using these true and predicted values.

    The mAP score is saved to a file at the end of the function.
    """
    # Get number of label files
    label_file_names = list_file_names(shared_constants.LABELS_DIR)
    num_files = len(label_file_names)
    logging.info(f"Number of label files: {num_files}")

    true_items = []
    predicted_items = []

    # Process each label and transform file
    for i in range(num_files):
        label_file_name = f'label_{i}.json'
        transform_file_name = f'img_{i}.jpg.out'

        # Get true label and bounding box
        with open(shared_constants.LABELS_DIR / label_file_name, "r") as f:
            _, _, true_label, true_x0, true_y0, true_x1, true_y1 = json.load(f)

        # Get predicted label and bounding box with highest confidence
        with open(shared_constants.TRANSFORM_DIR / transform_file_name, "r") as f:
            transform_contents = json.load(f)

        pred_label, confidence, pred_x0, pred_y0, pred_x1, pred_y1 = max(
            transform_contents["prediction"],
            key=lambda x: x[1]
        )

        # Skip if confidence is below threshold
        if confidence < CONF_THRESHOLD:
            continue

        # Add true and predicted items to lists
        true_items.append((int(true_label), true_x0, true_y0, true_x1, true_y1))
        predicted_items.append((int(pred_label), pred_x0, pred_y0, pred_x1, pred_y1))

    logging.debug(f"true_items={true_items}")
    logging.debug(f"predicted_items={predicted_items}")

    # Compute mAP score
    map_score = compute_mean_average_precision(true_boxes=true_items,
                                               predicted_boxes=predicted_items)
    logging.info(f"Obtained mAP score: {map_score}")

    # Save mAP score to file
    save_mean_average_precision_score(map_score)

    
def list_file_names(directory: pathlib.Path):
    return [str(file) for file in directory.glob('*') if file.is_file()]
    

def compute_mean_average_precision(
    true_boxes: List[List[float]],
    predicted_boxes: List[List[float]],
    iou_threshold: float = 0.5
) -> float:
    """Computes the mean average precision (mAP) for object detection.
    https://stackoverflow.com/a/47873723
    
    Args:
        true_boxes: A list of ground truth bounding boxes and their labels.
        predicted_boxes: A list of predicted bounding boxes, their labels and confidence scores.
        iou_threshold: The intersection over union threshold for a true positive.
    
    Returns:
        The mean average precision (mAP) score.
    """
    
    # Get unique labels
    unique_labels = sorted(list(set([box[0] for box in true_boxes])))
    
    # Initialize true positive and false positive counts
    true_positives = [0] * len(unique_labels)
    false_positives = [0] * len(unique_labels)
    
    # Process each label
    for label_index, label in enumerate(unique_labels):
        # Get ground truth and predicted boxes for current label
        true_label_boxes = [box[1:] for box in true_boxes if box[0] == label]
        predicted_label_boxes = [box[1:] for box in predicted_boxes if box[0] == label]
        
        num_predictions = len(predicted_label_boxes)
        
        # Skip if no predictions for current label
        if num_predictions == 0:
            continue

        # Initialize true positive and false positive arrays
        tp = [0] * num_predictions
        fp = [0] * num_predictions

        # Process each predicted box
        for i in range(num_predictions):
            # Get ground truth box if it exists, otherwise use None
            true_box = true_label_boxes[i] if i < len(true_label_boxes) else None
            
            # Compute intersection over union
            iou = compute_intersection_over_union(box1=predicted_label_boxes[i], box2=true_box)
            
            # Update true positive and false positive arrays
            if iou > iou_threshold:
                tp[i] = 1
            else:
                fp[i] = 1

        # Update true positive and false positive counts
        true_positives[label_index] = sum(tp)
        false_positives[label_index] = sum(fp)
    
    logging.debug(f"true_positives={true_positives}")
    logging.debug(f"false_positives={false_positives}")

    # Compute mean average precision (mAP)
    map_sum = 0.0
    for label_index, _ in enumerate(unique_labels):
        if true_positives[label_index] == 0.0:
            map_sum += 0
        else:
            map_sum += (
                true_positives[label_index]
                / (true_positives[label_index] + false_positives[label_index])
            )
    
    mean_average_precision = map_sum / float(len(unique_labels))
    
    return round(mean_average_precision, 4)


def compute_intersection_over_union(
    box1: Tuple[float, float, float, float],
    box2: Union[Tuple[float, float, float, float], None]
) -> float:
    """Computes the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: A tuple of the form (x_min, y_min, x_max, y_max) representing the first bounding box.
        box2: A tuple of the form (x_min, y_min, x_max, y_max) representing the second bounding box.

    Returns:
        The IoU of the two bounding boxes.
    """
    
    # Return 0 if second box is None
    if box2 is None:
        return 0.0
    
    # Compute intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute areas of both boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute union area
    union = area_box1 + area_box2 - intersection

    # Compute intersection over union
    iou = intersection / union

    return iou


def save_mean_average_precision_score(mAP):
    os.makedirs(shared_constants.EVALUATION_DIR, exist_ok=True)
    evaluate_file_path = os.path.join(shared_constants.EVALUATION_DIR, "evaluate.json")
    logging.info(f"Saving map to {evaluate_file_path}")
    with open(evaluate_file_path, "w+") as f:
        json.dump({"map": mAP}, f)
    logging.info(f"Successful save of map to {evaluate_file_path}")


if __name__ == "__main__":
    main()
