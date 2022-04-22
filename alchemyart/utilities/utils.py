import shutil
from pathlib import Path


def copy_subfolder_files(source: Path, destination: Path):
    c = 0
    for p in source.rglob("*"):
        if p.is_file():
            c += 1
            shutil.copy(p, destination)
    print(f"{c} files copied")


if __name__ == "__main__":
    source = Path("D:/datasets/AISegment_test/clip_img")
    destination = Path("D:/datasets/AISegment_test/x")
    copy_subfolder_files(source, destination)
