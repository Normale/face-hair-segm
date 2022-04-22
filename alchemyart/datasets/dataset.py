import cv2
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from imantics import Mask
from abc import ABCMeta, abstractmethod

import alchemyart.processing.processing as preprocessing
from alchemyart.datasets.common import *
from alchemyart.datasets.json_encoders import EnhancedJSONEncoder


class Dataset(metaclass=ABCMeta):
    def __init__(
        self, info: dict = {"kupa": "dupa"}, licenses: dict = {"X": "D"}
    ) -> None:
        self.info = info
        self.licenses = licenses

    @abstractmethod
    def generate_annotations(self):
        """
        Returns annotations to images in COCO format.
        """


class RawImagesDataset(Dataset):
    def __init__(
        self,
        x_folder: Path,
        y_folder: Path,
        info: dict = {"kupa": "dupa"},
        licenses: dict = {"X": "D"},
    ) -> None:
        super().__init__(info, licenses)
        self.x_folder = x_folder
        self.y_folder = y_folder

    def generate_annotations(self, format: Format = Format.COCO):
        if format == Format.COCO:
            return self.get_coco()
        else:
            raise NotImplementedError

    def save_annotations(self, filepath: Path, format: Format = Format.COCO):
        # there is always a parent, for filename its '.'
        parent_dir = filepath.parents[0]
        if not parent_dir.exists():
            print(f"Creating directory {parent_dir}")
            parent_dir.mkdir(parents=True, exist_ok=True)

        annotations = self.generate_annotations(format)
        with open(filepath, "w") as f:
            json.dump(annotations, f, cls=EnhancedJSONEncoder)

    def get_coco(self):
        print(
            f"calling generate coco with \nsrc:  {self.x_folder}"
            f"\nmasked:  {self.y_folder}"
        )
        coco = CoCoDict(self.info, self.licenses, [], [Category("person", 0, "person")])
        for i, img_path in enumerate(
            tqdm(self.x_folder.iterdir(), total=len(list(self.x_folder.iterdir())))
        ):
            name = img_path.name
            segmented_path = Path.joinpath(
                self.y_folder, img_path.name.replace(".jpg", ".png")
            )
            img = cv2.imread(str(img_path))
            h, w, *_ = img.shape
            segmented = cv2.imread(str(segmented_path), cv2.IMREAD_UNCHANGED)
            _, _, segmentations, bbox, area = preprocessing.get_annotations(segmented)

            annotation = Annotation(i, i, 0, 0, segmentations, bbox, area, "")
            coco_img = COCOImage(
                i, file_name=str(img_path), width=w, height=h, annotations=[annotation]
            )
            coco.images.append(coco_img)
        return coco

    def preview_images(self, image_substr: str = "", mode: str = "greenscreen"):
        for img_path in self.x_folder.iterdir():
            if not image_substr in str(img_path):
                continue
            segmented_path = Path.joinpath(
                self.y_folder, img_path.name.replace(".jpg", ".png")
            )
            img = cv2.imread(str(img_path))
            segmented = cv2.imread(str(segmented_path), cv2.IMREAD_UNCHANGED)
            person = segmented[:, :, 3] == 0

            if mode == "greenscreen":
                masked = np.copy(img)
                masked[person] = (0, 255, 0)
            if mode == "polygons":
                person = np.logical_not(person)
                polygons = Mask(person).polygons()
                masked = polygons.draw(img, color=(255, 0, 0))

            cv2.imshow("Original", img)
            cv2.imshow("Segmented", segmented)
            cv2.imshow("Masked", masked)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
