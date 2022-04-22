from typing import Tuple
from imantics import Mask
import numpy as np


def get_polygons(mask: np.ndarray):
    return Mask(mask).polygons()


def get_transparent(img: np.ndarray):
    return img[:, :, 3] != 0


def get_bbox(coords: np.ndarray):
    px = coords[::2]
    py = coords[1::2]
    return np.min(px), np.min(py), np.max(px), np.max(py)


def get_annotations(img: np.ndarray) -> Tuple:
    mask = get_transparent(img)
    polygons = get_polygons(mask)
    segmentations = polygons.segmentation
    segmentations = list(filter(lambda x: len(x) > 5, segmentations))
    bbox_tmp = polygons.bbox()
    bbox = bbox_tmp.top_left + bbox_tmp.size  # (X, Y, W, H)
    # ! ASSUMPTION: THERE IS EXACTLY ONE OBJECT.
    area = mask.sum()

    return mask, polygons, segmentations, bbox, area
