
import os

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
openai_api_key = os.getenv('OPENAI_API_KEY')

#sotrage root point
storage_root="./storage/"


# Phase tracking
current_phase = None  # 'drive' or 'recovery'
drive_start_frame = None
recovery_start_frame = None

# Storing times
drive_times = []
recovery_times = []

# Video Frame Rate
frame_rate = 30.0  # Adjust this according to your video's frame rate


# Configuration for image processing
IMAGE_STRIDE = 64
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.65

# Configuration for drawing and keypoints
LINE_THICKNESS = 3
KEYPOINT_STEPS = 3

# Indexes for keypoints
RIGHT_ELBOW_INDEX = 8 * KEYPOINT_STEPS
LEFT_ELBOW_INDEX = 7 * KEYPOINT_STEPS
RIGHT_SHOULDER_INDEX = 6 * KEYPOINT_STEPS
LEFT_SHOULDER_INDEX = 5 * KEYPOINT_STEPS
RIGHT_HIP_INDEX = 12 * KEYPOINT_STEPS
LEFT_HIP_INDEX = 11 * KEYPOINT_STEPS
RIGHT_KNEE_INDEX = 14 * KEYPOINT_STEPS
LEFT_KNEE_INDEX = 13 * KEYPOINT_STEPS
RIGHT_ANKLE_INDEX = 16 * KEYPOINT_STEPS
LEFT_ANKLE_INDEX = 15 * KEYPOINT_STEPS
RIGHT_WRIST_INDEX = 10 * KEYPOINT_STEPS
LEFT_WRIST_INDEX = 9 * KEYPOINT_STEPS
# Previous wrist Y position for phase detection
prev_wrist_y = None
