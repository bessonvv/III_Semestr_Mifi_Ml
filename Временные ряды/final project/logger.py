import os
from datetime import datetime
from config import LOG_FILE
from typing import Dict


def log_request(
    user_id: int,
    ticker: str,
    investment_amount: float,
    model_name: str,
    metric_value: float,
    profit: float
) -> None:
    """
    Записывает информацию о запросе в лог-файл.
    
    Args:
        user_id: ID пользователя Telegram
        ticker: Тикер компании
        investment_amount: Сумма инвестиции
        model_name: Название выбранной модели
        metric_value: Значение метрики RMSE
        profit: Рассчитанная потенциальная прибыль
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    log_entry = (
        f"{timestamp} | "
        f"UserID: {user_id} | "
        f"Ticker: {ticker} | "
        f"Investment: ${investment_amount:.2f} | "
        f"Model: {model_name} | "
        f"RMSE: {metric_value:.2f} | "
        f"Profit: ${profit:.2f}\n"
    )
    
    # Проверяем, нужен ли заголовок (если файл новый)
    is_new_file = not os.path.exists(LOG_FILE)
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        if is_new_file:
            header = "Timestamp | UserID | Ticker | Investment | Model | RMSE | Profit\n"
            f.write(header)
            f.write("=" * 80 + "\n")
        f.write(log_entry)


def create_log_header_if_needed() -> None:
    """Создаёт заголовок в лог-файле, если его нет."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("=== TELEGRAM BOT STOCK FORECAST LOG ===\n")
            f.write("Timestamp | UserID | Ticker | Investment | Model | RMSE | Profit\n")
            f.write("=" * 80 + "\n")
