import cv2
import torch
import os
import numpy as np
import requests
from urllib.parse import urlparse
from pytube import YouTube
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
# Importing necessary utilities from a hypothetical utils module - adjust imports based on your project structure
from utils.general import non_max_suppression_kpt, scale_coords, xyxy2xywh, strip_optimizer, check_img_size
from utils.plots import plot_one_box_kpt, plot_skeleton_kpts, colors, output_to_keypoint
from utils.datasets import letterbox
from utils.torch_utils import select_device, time_synchronized
from models.experimental import attempt_load
from utilities import is_valid_url, get_content_type
from frame_processor import process_frame 
import config

# variabke to keeo previous wrist position

# Assuming utils.py and necessary model files are correctly set up in your project directory

def download_youtube_video(url, download_folder="{config.storage_root}/downloaded_videos"):
    Path(download_folder).mkdir(parents=True, exist_ok=True)
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if not stream:
        print("Suitable video stream not found.")
        return None
    video_path = stream.download(output_path=download_folder)
    print(f"Video downloaded successfully: {video_path}")
    return video_path

def download_video_from_url(url, download_folder="downloaded_videos"):
    if not is_valid_url(url):
        print("The URL is not valid or secure.")
        return None
    content_type = get_content_type(url)
    if content_type and "video" in content_type:
        file_name = os.path.basename(urlparse(url).path)
        file_path = os.path.join(download_folder, file_name)
        try:
            os.makedirs(download_folder, exist_ok=True)
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
            print(f"Video downloaded successfully: {file_path}")
            return file_path
        except requests.exceptions.RequestException as e:
            print(f"Failed to download the video: {e}")
            return None
    else:
        print("The URL does not point to a video content.")
        return None

@torch.no_grad()
def process_video_analysis(source="input_video.mp4", poseweights="yolov7-w6-pose.pt", device='cpu', view_img=False, target_fps=1, save_conf=False, line_thickness=3, hide_labels=False, hide_conf=True):
    #device = select_device(device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    half = device.type != 'cpu'
    half=False
    model = attempt_load(poseweights, map_location=device).eval()
    if half:
        model.half()  # To FP16
    stride = int(model.stride.max())  # Model stride
    names = model.module.names if hasattr(model, 'module') else model.names

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip_interval = int(np.round(source_fps / target_fps))
    output_dir= f"{config.storage_root}/processed_videos"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_video_path = Path(output_dir) / f"{Path(source).stem}_keypoint.mp4"
    vid_write_image = letterbox(cap.read()[1], stride=64, auto=True)[0]  # Init VideoWriter
    resize_height, resize_width = vid_write_image.shape [:2]
    # need to fix the size out the output cideo writer
    out = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (resize_width*2, resize_height*2))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Current frame position
        if frame_count % frame_skip_interval == 0:
            processed_frame = process_frame(frame, model, device, names, hide_labels, hide_conf)      
            if view_img:
                cv2.imshow('Processed Frame', processed_frame)
                if cv2.waitKey(1) == ord('q'):
                    break
#            print("Processed frame type:", type(processed_frame))
#            print("Processed frame shape:", processed_frame.shape if processed_frame is not None else "None")
            out.write(processed_frame)  # Write processed frame
    

    cap.release()
    out.release()
#    cv2.destroyAllWindows()
    
    return output_video_path



# Placeholder for additional utility functions as needed based on the provided logic
# Implement or import these as necessary for your project

if __name__ == "__main__":
    # Example usage
    process_video_analysis(source="test2.mp4", poseweights="yolov7-w6-pose.pt", device='gpu', view_img=False,target_fps=3)
