import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def create_forecast_plot(
    df: pd.DataFrame, 
    forecast: np.ndarray, 
    ticker: str,
    save_path: str = 'forecast_plot.png'
) -> str:
    """
    Создаёт график с историческими данными и прогнозом.
    
    Args:
        df: DataFrame с историческими данными
        forecast: Массив с прогнозными значениями
        ticker: Тикер компании
        save_path: Путь для сохранения графика
    
    Returns:
        Путь к сохранённому файлу
    """
    plt.figure(figsize=(14, 7))
    
    # Исторические данные (последние 90 дней для наглядности)
    hist_df = df.tail(90).copy()
    plt.plot(hist_df['Date'], hist_df['Price'], 
             label='Исторические данные', color='blue', linewidth=2)
    
    # Прогнозные данные
    last_date = df['Date'].iloc[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast))]
    
    plt.plot(forecast_dates, forecast, 
             label='Прогноз', color='red', linewidth=2, linestyle='--')
    
    # Точка перехода от истории к прогнозу
    plt.scatter([last_date], [df['Price'].iloc[-1]], 
                color='green', s=100, zorder=5, label='Текущая цена')
    
    # Оформление
    plt.title(f'Прогноз цены акций {ticker}', fontsize=16, fontweight='bold')
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Цена ($)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Форматирование дат на оси X
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def format_price_change(current_price: float, forecast: np.ndarray) -> str:
    """
    Форматирует информацию об изменении цены.
    
    Args:
        current_price: Текущая цена акции
        forecast: Массив прогнозных значений
    
    Returns:
        Строка с описанием изменения
    """
    final_price = forecast[-1]
    price_change = final_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    direction = "вырастут" if price_change > 0 else "упадут"
    
    message = f"""
📊 Прогноз изменения цены:

Текущая цена: ${current_price:.2f}
Прогноз через 30 дней: ${final_price:.2f}

Изменение: ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%)
Направление: Акции {direction}
"""
    
    return message
