from enum import Enum
from dataclasses import dataclass, field
from typing import Any, List, Union


class Format(Enum):
    """
    Dataset subsets.
    """

    COCO = 1
    DETECTRON = 2


@dataclass
class Annotation:
    id: int
    image_id: int = 0  # no idea how does it differ from `id`
    category_id: int = ""
    iscrowd: int = ""  # 0 or 1, crowd is 1 mask over multiple instances
    segmentation: List[List[float]] = field(default_factory=list)
    # it can also be dict
    # segmentation: Union[dict, List[List[float]]]= {
    #     "size": [3,3],
    #     "counts": [4, 1, 4]} # for 3x3 with mid px "segmented"
    bbox: Union[float, int] = 0
    area: Union[float, int] = ""
    date_captured: str = ""


@dataclass
class COCOImage:
    id: int
    license: str = ""
    coco_url: str = ""
    flickr_url: str = ""
    width: int = 0
    height: int = 0
    file_name: str = ""
    date_captured: Any = ""
    annotations: List[Annotation] = field(default_factory=list)


@dataclass
class Category:
    supercategory: str = ""
    id: int = ""
    name: str = ""


@dataclass
class CoCoDict:
    info: dict
    licenses: dict
    images: List[int]
    # annotations: List[Annotation]
    categories: List[Category]
