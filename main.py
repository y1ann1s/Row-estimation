# main.py
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import logging
from bot_handlers import start, help_command, handle_file, handle_message, handle_video, error_handler
from config import TELEGRAM_TOKEN
from flask_app import run_app  # Import the Flask app running function h
from threading import Thread

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def main() -> None:
    # Initialize Telegram Bot
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.Document.TEXT, handle_file))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    #application.add_handler(MessageHandler(filters.Document.MIME_TYPE('text/csv') | filters.Document.MIME_TYPE('video/mp4'), handle_file))

    # Run the Flask app on a different thread
    #flask_thread = Thread(target=run_app)
    #flask_thread.start()

    # Run Telegram Bot
    application.run_polling()

if __name__ == '__main__':
    main()
