import shutil
from pathlib import Path
import numpy as np
from typing import Tuple
from itertools import chain
import shutil
from tqdm.notebook import tqdm
import cv2

def copy_subfolder_files(source: Path, destination: Path):
    if not destination.exists():
        print(f"Creating directory {destination}")
        destination.mkdir(parents=True, exist_ok=True)

    c = 0
    for p in source.rglob("*"):
        if p.is_file():
            c += 1
            shutil.copy(p, destination)
    print(f"{c} files copied")


def split_dataset(proportions: Tuple[float], source_x: Path, source_y: Path, x_paths: Tuple[Path], y_paths: Tuple[Path]):
    '''
        Splits images in source folders to 3 folders: train, val, test.

        :param proportions: Tuple of 3 values representing proportions of sizes of datasets. 
            Example (0.7, 0.2, 0.1) for 70% training, 20% val, 10% test
    '''
    for path in chain(x_paths, y_paths):
        path.mkdir(parents=True, exist_ok=True)
    assert sum(proportions) == 1, "Proportions tuple must add up to 1"
    assert len(proportions) == len(x_paths) == len(y_paths)
    no_splits = len(x_paths)
    all_files = np.array(list(source_x.iterdir()))
    np.random.shuffle(all_files)
    borders = [round(x) for x in np.cumsum(proportions) * len(all_files)]
    filenames_split = np.split(all_files,borders)
    for i in tqdm(range(no_splits)):
        folder_x_path = x_paths[i]
        folder_y_path = y_paths[i]
        for x_item in tqdm(filenames_split[i]):
            y_name = x_item.name.replace('.jpg', '.png')
            y_item = source_y / y_name
            if not y_item.exists():
                raise FileNotFoundError
            shutil.move(str(x_item), str(folder_x_path))
            shutil.move(str(y_item), str(folder_y_path))

def save_masks_as_black_white(src, dest):
    dest.mkdir(parents=True, exist_ok=True)
    all_files = np.array(list(src.iterdir()))
    for f in all_files:
        print(f)
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        person = img[:, :, 3] != 0
        person = person.astype(np.uint8)
        person *=255
        cv2.imwrite(str(dest / f.name).replace(".png", ".jpg"), person.astype(np.uint8))


if __name__ == "__main__":
    # source = Path("D:/datasets/AISegment/clip_img")
    # destination = Path("D:/datasets/AISegment_whole/x")
    # copy_subfolder_files(source, destination)

    # source = Path("D:/datasets/AISegment/matting")
    # destination = Path("D:/datasets/AISegment_whole/y")
    # copy_subfolder_files(source, destination)

    # src_x = Path(r"D:\datasets\AISegment_test\x")
    src_y = Path(r"D:\datasets\AISegment_test\y")
    # general = Path(r"D:\datasets\AISegment_test\split\x")
    # x_paths = (general / "train", general / "val", general / "test")
    # general = Path(r"D:\datasets\AISegment_test\split\y")
    # y_paths = (general / "train", general / "val", general / "test")
    # split_dataset((0.5,0.3,0.2),src_x, src_y, x_paths, y_paths)

    dest_black = Path(r"D:\datasets\AISegment_test\y_black")
    save_masks_as_black_white(src_y, dest_black)