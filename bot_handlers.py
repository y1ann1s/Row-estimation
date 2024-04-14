from telegram import Update
from telegram.ext import ContextTypes
import telegram
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from video_processing import download_youtube_video, process_video_analysis, download_video_from_url
from openai_chat import ask_chatgpt
import logging

logger = logging.getLogger(__name__)
URL_REGEX = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
executor = ThreadPoolExecutor(max_workers=4)

async def process_video_and_respond(path, update, context):
    loop = asyncio.get_running_loop()
    try:
        result, video_path = await loop.run_in_executor(executor, process_video_analysis, path)
        await update.message.reply_text(f"Analysis Complete: {result}")
        await context.bot.send_video(chat_id=update.effective_chat.id, video=open(video_path, 'rb'))
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        await update.message.reply_text("Failed to process the video. Please try again later.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message.text:
        urls = re.findall(URL_REGEX, message.text)
        if not urls:  # No URLs found, process text through ChatGPT
            chat_history = context.chat_data.get('chat_history', [])  # Retrieve chat history as a list
                       
            response, updated_history = ask_chatgpt(message.text, chat_history)
            #response = ask_chatgpt(message.text, chat_history)
            
            context.chat_data['chat_history'] = updated_history  # Update chat history in context
            await message.reply_text(response)
            return
        # Handle URLs found in message text
        for url in urls:
            if "youtube.com" in url or "youtu.be" in url:
                video_path = download_youtube_video(url)
            else:
                video_path = download_video_from_url(url)
            if video_path:
                await message.reply_text("Processing your video, please wait...")
                await process_video_and_respond(video_path, update, context)
                return  # Exit after processing the first valid URL
            else:
                await message.reply_text("Failed to download the video. Please check the URL and try again.")
    else:
        await message.reply_text("Please send a video URL to analyze or ask a question about rowing.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hi! I am your rowing coach. Send me a video URL to start the analysis or ask me anything about rowing!')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Send a video URL, and I will analyze it for you. You can also ask questions about rowing techniques!')

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message.video:
        video_file = await context.bot.get_file(message.video.file_id)
        video_file_path = f"{message.video.file_id}.mp4"  # Define file path
        await video_file.download(video_file_path)
        logger.info(f"Video file downloaded: {video_file_path}")
        await message.reply_text("Processing your video, please wait...")
        await process_video_and_respond(video_file_path, update, context)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors occurred within the bot."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    if isinstance(context.error, telegram.error.BadRequest) and "File is too big" in str(context.error):
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text="Sorry, the video file you sent is too large for me to process.")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text="An unexpected error occurred. Please try again later.")
