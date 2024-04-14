from telegram.ext import Application, CommandHandler, MessageHandler, filters
import logging
from bot_handlers import start, help_command, handle_video, handle_message, error_handler
from config import TELEGRAM_TOKEN

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    # Handle video uploads
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    # Handle text messages (for URLs)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    application.run_polling()

if __name__ == '__main__':
    main()
