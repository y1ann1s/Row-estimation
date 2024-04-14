from telegram import Update, ForceReply
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import telegram
import logging
import nest_asyncio
import cv2
import numpy as np
import os
import re
import requests
import mimetypes
from urllib.parse import urlparse, urljoin
from urllib.request import urlopen
from pytube import YouTube
import cv2
import numpy as np
import torch
from torchvision import transforms
from utils.plots import plot_one_box_kpt
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt
from utils.plots import plot_one_box_kpt
import re
from telegram import Update
from telegram.ext import ContextTypes
import logging

#from google.colab import drive
#drive.mount('/content/drive')
nest_asyncio.apply()

# Enable logging for debugging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("app.log"),  # Log to this file
                        logging.StreamHandler()  # Log to standard output (console)
                    ])
logger = logging.getLogger(__name__)


def download_youtube_video(url, download_folder="downloaded_videos"):
    """
    Downloads a video from YouTube using pytube.

    Args:
        url (str): The URL of the YouTube video.
        download_folder (str): Folder where the video will be saved.

    Returns:
        str: The file path to the downloaded video.
    """
    try:
        # Create the download folder if it doesn't exist
        os.makedirs(download_folder, exist_ok=True)

        # Use pytube to download the video
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if not stream:
            print("Suitable video stream not found.")
            return None

        # Download the video
        video_path = stream.download(output_path=download_folder)
        print(f"Video downloaded successfully: {video_path}")
        return video_path
    except Exception as e:
        print(f"Failed to download YouTube video: {e}")
        return None


def is_valid_url(url):
    """
    Checks if the provided URL is valid and secure.
    """
    parsed_url = urlparse(url)
    # Ensure scheme is either HTTP or HTTPS
    if parsed_url.scheme not in ("http", "https"):
        return False
    # Check if the domain is considered safe. This list can be extended.
    safe_domains = ["youtube.com", "vimeo.com", "example.com"]
    if parsed_url.netloc not in safe_domains:
        return False
    return True

def get_content_type(url):
    """
    Returns the Content-Type of the file located at the URL.
    """
    try:
        response = urlopen(url)
        # Only check the header part for the content type
        content_type = response.info().get_content_type()
        return content_type
    except Exception as e:
        print(f"Failed to get content type for URL {url}: {e}")
        return None

def download_video_from_url(url, download_folder="downloaded_videos"):
    """
    Downloads the video from the URL after performing security checks and verifying MIME type.
    """
    if not is_valid_url(url):
        print("The URL is not valid or secure.")
        return None

    content_type = get_content_type(url)
    if content_type and "video" in content_type:
        # Construct a safe file name from the URL
        file_name = os.path.basename(urlparse(url).path)
        file_path = os.path.join(download_folder, file_name)

        try:
            os.makedirs(download_folder, exist_ok=True)
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error on bad status

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Video downloaded successfully: {file_path}")
            return file_path
        except requests.exceptions.RequestException as e:
            print(f"Failed to download the video: {e}")
    else:
        print("The URL does not point to a video content.")
    return None


# Regular expression to find URLs in a text message
URL_REGEX = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def is_video_url(url):
    # Placeholder for checking if the URL points to a video file
    # This might involve checking the MIME type or file extension in a real scenario
    parsed_url = urlparse(url)
    return parsed_url.path.endswith(('.mp4', '.avi', '.mov'))


# Assuming all necessary functions are defined or imported:
# - is_video_url, download_video_from_url, download_youtube_video, process_video_analysis

logger = logging.getLogger(__name__)

URL_REGEX = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles both direct video uploads and video URLs (including YouTube)."""
    message = update.message

    if message.video:
        # Direct video upload case
        video_file = await context.bot.get_file(message.video.file_id)
        video_file_path = f"{message.video.file_id}.mp4"  # Define file path
        await video_file.download(video_file_path)
        logger.info(f"Video file downloaded: {video_file_path}")

        # Process the downloaded video
        analysis_result = process_video_analysis(video_file_path)
        await message.reply_text(f"Analysis Complete: {analysis_result}")
    elif message.text:
        # URL case
        urls = re.findall(URL_REGEX, message.text)
        video_url = None
        for url in urls:
            if "youtube.com" in url or "youtu.be" in url:
                # YouTube URL case
                video_path = download_youtube_video(url)
                if video_path:
                    analysis_result = process_video_analysis(video_path)
                    await message.reply_text(f"Analysis Complete: {analysis_result}")
                    return
            elif is_video_url(url):
                video_url = url
                break

        if video_url:
            # Other video URL case
            video_path = download_video_from_url(video_url)
            if video_path:
                analysis_result = process_video_analysis(video_path)
                await message.reply_text(f"Analysis Complete: {analysis_result}")
            else:
                await message.reply_text("Failed to download the video from the provided URL.")
        elif not urls:
            # No URLs detected in message
            await message.reply_text("Please send a direct video file or a video URL.")



def process_video_analysis(video_path, poseweights="yolov7-w6-pose.pt", device='cpu', output_dir="processed_videos"):
    device = torch.device(device)
    model = attempt_load(poseweights, map_location=device)  # Load model
    model.eval()  # Set model to evaluation mode

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Failed to open video file."

    # Define output video path
    output_video_path = os.path.join(output_dir, os.path.basename(video_path))
    os.makedirs(output_dir, exist_ok=True)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame for pose estimation and plotting
        frame = process_frame(frame, model, device)

        # Write the frame with keypoints and lines into the output video
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_video_path, "Analysis complete. Check out the video with overlaid keypoints."


def load_pose_estimation_model():
    # Placeholder for model loading logic
    # Replace this with your actual model loading code
    return "Your Pose Estimation Model"

def pose_estimation(frame, model):
    # Placeholder for pose estimation logic
    # In reality, you would process the frame through your model here
    # and return detected poses
    return {"pose_keypoints": "Detected Pose Keypoints"}

def analyze_frame(poses):
    # Placeholder for frame analysis logic based on detected poses
    # Here you would analyze the pose to determine the rowing phase
    # and other technique metrics
    return "Frame Analysis Result"

def summarize_analysis(analysis_results, frame_count):
    # Compile the frame-by-frame analysis into a comprehensive summary
    # This could include counting occurrences of each rowing phase,
    # identifying technique errors, and providing overall feedback
    return f"Analysis Summary based on {frame_count} frames"

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors occurred within the bot."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    # Here you can add specific error handling logic
    if isinstance(context.error, telegram.error.BadRequest) and "File is too big" in str(context.error):
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text="Sorry, the video file you sent is too large for me to process.")
    else:
        # Handle other exceptions or send a generic error message
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text="An unexpected error occurred. Please try again later.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text('Hi! Send me a video to analyze your rowing technique.')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text('Send a video, and I will analyze your rowing technique.')

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming video messages."""
    user = update.message.from_user
    logger.info(f"Received video from {user.first_name} (username: {user.username})")

    # Define the directory in Google Drive to save the video
    save_directory = '/content/drive/My Drive/Colab Notebooks/VideoAnalysis'
    os.makedirs(save_directory, exist_ok=True)  # Ensure the directory exists

    # Define the local filename to save the video
    video_file_path = os.path.join(save_directory, f"{update.message.video.file_id}.mp4")

    # Get the video file from the message
    video_file = await context.bot.get_file(update.message.video.file_id)

    # Download the video file
    await video_file.download(video_file_path)
    logger.info(f"Video file downloaded to {video_file_path}")

    # Now you can process the video and send back the analysis
    # Remember to adjust the process_video_analysis function to handle the video path correctly
    analysis_result = process_video_analysis(video_file_path)

    # Send the analysis result back to the user
    await update.message.reply_text(f"Analysis Complete: {analysis_result}")


def calculate_angle(pt1, pt2, pt3):
    """
    Calculate the angle (in degrees) formed by three points.

    Args:
        pt1, pt2, pt3 (tuple): Points in the format (x, y).

    Returns:
        float: The angle in degrees.
    """
    # Vector 1 (pt2 to pt1) and Vector 2 (pt2 to pt3)
    vector1 = np.array([pt1[0] - pt2[0], pt1[1] - pt2[1]])
    vector2 = np.array([pt3[0] - pt2[0], pt3[1] - pt2[1]])

    # Calculate angle in radians
    angle_rad = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Assuming video download and initial processing have been done
    video_path, message = process_video_analysis(downloaded_video_path)
    if video_path:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=message)

        # Send the resulting video back to the user
        await context.bot.send_video(chat_id=update.effective_chat.id, video=open(video_path, 'rb'))
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Failed to process the video.")

def process_frame(frame, model, device):
    # Previous frame processing steps...

    # Example: Using dummy indices for keypoints, replace with your model's specifics
    hip_index = 0
    knee_index = 1
    ankle_index = 2
    shoulder_index = 3

    for pose in preds:
        keypoints = pose['keypoints']
         # Plot keypoints on the frame - this is simplified, and you might need adjustments
        # Here, we assume 'plot_one_box_kpt' or a similar utility function is used for plotting
        for kpt in keypoints:
            x, y, conf = kpt[:3]
            if conf > 0.5:  # Confidence threshold
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Optionally, draw lines between keypoints based on your model's skeleton representation
        # This would require a mapping from keypoints to body parts and knowing which parts connect
        # For example, if keypoints[0] and keypoints[1] are connected:
        if len(keypoints) > 1:
            cv2.line(frame, (int(keypoints[0][0]), int(keypoints[0][1])),
                           (int(keypoints[1][0]), int(keypoints[1][1])), (255, 0, 0), 2)

        # Example calculations, ensure keypoints are present and confidence is high enough
        if len(keypoints) > max(hip_index, knee_index, ankle_index, shoulder_index):
            # Calculate angles
            knee_angle = calculate_angle(keypoints[hip_index][:2], keypoints[knee_index][:2], keypoints[ankle_index][:2])
            shin_angle = calculate_angle(keypoints[knee_index][:2], keypoints[ankle_index][:2], (keypoints[ankle_index][0], keypoints[ankle_index][1]+10))  # Assuming vertical shin as reference
            torso_angle = calculate_angle(keypoints[shoulder_index][:2], keypoints[hip_index][:2], (keypoints[hip_index][0], keypoints[hip_index][1]+10))  # Assuming vertical torso as reference

            # Overlay angle information on the frame
            cv2.putText(frame, f"Knee Angle: {int(knee_angle)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Shin Angle: {int(shin_angle)}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Torso Angle: {int(torso_angle)}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Return the processed frame
    return frame

def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token("6741348449:AAEJwzvHxhyvZdI1kz_fPWTNLR5AAkl5CnU").build()

    # Register handlers in the application
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_error_handler(error_handler)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()
