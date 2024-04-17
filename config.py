
import os

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
openai_api_key = os.getenv('OPENAI_API_KEY')

#sotrage root point
storage_root="./storage/"

# Other global configuration variables can go here
prev_wrist_y = None


