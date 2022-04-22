import cv2
import numpy as np
import glob
from pathlib import Path 


def read_images(folder: Path):
    '''
    Assumption: files are in format: f"{number}.{ext}"
    '''
    result = []
    images = [x for x in folder.iterdir() if x.is_file()]
    filelist = sorted(images,key=lambda x: int(x.stem))
    for img_name in filelist:
        print(img_name)
        img_path = folder / img_name
        img = cv2.imread(str(img_path))
        result.append(img)
    return result

def create_video_from_frames(frames: list, save_filepath: Path): 
    save_filepath.parents[0].mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    size = (w,h)
    out = cv2.VideoWriter(str(save_filepath),cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()


def save_images_from_video(video_path: Path, save_folder: Path):
    # Read the video from specified path
    cam = cv2.VideoCapture(str(video_path))
    save_folder.mkdir(parents=True, exist_ok=True)
    
    currentframe = 0
    while(True):
        print(f"frame {currentframe}")
        ret,frame = cam.read()
        if ret:
            # if video is still left continue creating images
            name = save_folder / f"{currentframe}.png"
            print(name)
            # writing the extracted images
            cv2.imwrite(str(name), frame)
            currentframe += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = Path(r"C:\Users\bartek\Desktop\GitHub\face-hair-segm\demo\data\NO.mp4")
    save_path = Path(r"C:\Users\bartek\Desktop\GitHub\face-hair-segm\demo\data\images")
    video_return_path = Path(r"C:\Users\bartek\Desktop\GitHub\face-hair-segm\demo\data\videos\NO_recreated.mp4")
    # save_images_from_video(video_path, save_path)
    frames = read_images(save_path)
    create_video_from_frames(frames, video_return_path)