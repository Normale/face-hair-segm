from pathlib import Path
from alchemyart.datasets.dataset import RawImagesDataset

if __name__ == "__main__":
    general = Path("D:\\datasets\\AISegment")
    # general = Path("D:\\datasets\\AISegment_test")
    src = "clip_img\\1803151818\\clip_00000000"
    segmented = "matting\\1803151818\\matting_00000000"
    src_path = general / src
    segmented_path = general / segmented
    # generate_coco(src_path, segmented_path)

    # result = test_dataset(src_path, segmented_path)
    ds = RawImagesDataset(src_path, segmented_path)
    ds.preview_images("532", mode="polygons")
    ds.preview_images("532", mode="greenscreen")
    ds.preview_images(mode="greenscreen")
    ds.save_annotations(Path("D:/datasets/x/y/z/1803151818.00000000PATH.json"))

    print("the end.")
