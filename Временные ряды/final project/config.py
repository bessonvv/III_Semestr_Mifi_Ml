import os
from dotenv import load_dotenv

load_dotenv()

# Токен бота из переменной окружения или напрямую
BOT_TOKEN = os.getenv('BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')

# Параметры моделей
DATA_PERIOD = '2y'  # Период загрузки данных
FORECAST_DAYS = 30  # Количество дней для прогноза
TEST_SIZE = 0.2  # Размер тестовой выборки

# Параметры логирования
LOG_FILE = 'logs.txt'
