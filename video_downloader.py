from urllib.parse import urlparse
import requests
from pytube import YouTube
from telegram import Update
from telegram.ext import ContextTypes
import video_processing
import re

URL_REGEX = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_text = update.message.text
    urls = re.findall(URL_REGEX, message_text)
    for url in urls:
        if "youtube.com" in url or "youtu.be" in url:
            video_path = download_youtube_video(url)
            if video_path:
                analysis_result = video_processing.process_video_analysis(video_path)
                await update.message.reply_text(analysis_result)
                return
    # Additional handling for other URLs or indicating no valid URL found

def download_youtube_video(url, download_folder="downloaded_videos"):
    # YouTube video downloading logic
    return "path/to/downloaded/video.mp4"
