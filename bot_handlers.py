from telegram import Update
from telegram.ext import ContextTypes
import telegram
import re
import asyncio
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from video_processing import download_youtube_video, process_video_analysis, download_video_from_url
from openai_chat import ask_chatgpt
import logging
from pathlib import Path
import config

logger = logging.getLogger(__name__)
URL_REGEX = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
executor = ThreadPoolExecutor(max_workers=4)

async def process_video_and_respond(path, update, context):
    loop = asyncio.get_running_loop()
    try:
        result = "Success"
        video_path = await loop.run_in_executor(executor, process_video_analysis, path)
        await update.message.reply_text(f"Analysis Complete: {result}")
        await context.bot.send_video(chat_id=update.effective_chat.id, video=open(video_path, 'rb'))
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        await update.message.reply_text("Failed to process the video. Please try again later.")

async def process_csv_and_respond(file_path, update, context):
    try:
        df = pd.read_csv(file_path)
        csv_content = df.to_string()
        chat_history = context.chat_data.get('chat_history', [])
        response, updated_history = ask_chatgpt(csv_content, chat_history)
        context.chat_data['chat_history'] = updated_history
        await update.message.reply_text("Here's the analysis of your CSV file:")
        await update.message.reply_text(response)
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        await update.message.reply_text("Failed to process the CSV file. Please try again later.")
        
async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message.video:
        video_file = await context.bot.get_file(message.video.file_id)
        download_folder=f"{config.storage_root}/downloaded_video"
        Path(download_folder).mkdir(parents=True, exist_ok=True)
        video_file_path = f"{download_folder}/{message.from_user}_{message.video.file_id}.{message.video.mime_type.split('/')[-1]}"   # Define file path
        await video_file.download_to_drive(video_file_path)
        logger.info(f"Video file downloaded: {video_file_path}")
        await message.reply_text("Processing your video, please wait...")
        await process_video_and_respond(video_file_path, update, context)

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    file = await context.bot.get_file(message.document.file_id)
    download_folder=f"{config.storage_root}/downloaded_csv"
    Path(download_folder).mkdir(parents=True, exist_ok=True)
    file_path = f"{download_folder}/{message.from_user}_{message.document.file_id}.{message.document.mime_type.split('/')[-1]}"  # Preserves original file extension
       
    await file.download_to_drive(file_path)
    logger.info(f"File downloaded: {file_path}")

    if message.document.mime_type == 'text/csv':
        await update.message.reply_text("Processing your CSV file, please wait...")
        await process_csv_and_respond(file_path, update, context)
    elif "video" in message.document.mime_type:  # Handles video files
        await update.message.reply_text("Processing your video, please wait...")
        await process_video_and_respond(file_path, update, context)
    else:
        await update.message.reply_text("Unsupported file type.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message.document:
        await handle_file(update, context)
    elif message.text:
        urls = re.findall(URL_REGEX, message.text)
        if not urls:
            chat_history = context.chat_data.get('chat_history', [])
            response, updated_history = ask_chatgpt(message.text, chat_history)
            context.chat_data['chat_history'] = updated_history
            await message.reply_text(response)
            return
        for url in urls:
            if "youtube.com" in url or "youtu.be" in url:
                video_path = download_youtube_video(url)
            else:
                video_path = download_video_from_url(url)
            if video_path:
                await message.reply_text("Processing your video, please wait...")
                await process_video_and_respond(video_path, update, context)
                return
            else:
                await message.reply_text("Failed to download the video. Please check the URL and try again.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hi! I am your rowing coach. Send me a video URL to start the analysis or ask me anything about rowing!')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Send a video URL, and I will analyze it for you. You can also ask questions about rowing techniques!')

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    if isinstance(context.error, telegram.error.BadRequest) and "File is too big" in str(context.error):
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, the video file you sent is too large for me to process.")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="An unexpected error occurred. Please try again later.")
