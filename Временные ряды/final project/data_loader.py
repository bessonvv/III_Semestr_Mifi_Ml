import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from config import DATA_PERIOD
import time


def load_stock_data(ticker: str) -> tuple[pd.DataFrame, str]:
    """
    Загружает исторические данные акций за указанный период.
    
    Args:
        ticker: Тикер компании (например, AAPL, MSFT)
    
    Returns:
        DataFrame с данными и сообщение об ошибке (если есть)
    """
    try:
        # Используем download вместо Ticker().history() для большей надёжности
        print(f"Загрузка данных для {ticker}...")
        
        # Пробуем несколько методов загрузки
        df = None
        
        # Метод 1: yf.download с дополнительными параметрами
        try:
            df = yf.download(
                ticker, 
                period=DATA_PERIOD,
                progress=False
            )
            if not df.empty:
                print(f"Загружено {len(df)} записей через download")
        except Exception as e:
            print(f"Метод download не сработал: {e}")
        
        # Метод 2: Если download не сработал, пробуем через Ticker
        if df is None or df.empty:
            print("Пробую альтернативный метод...")
            time.sleep(1)  # Небольшая задержка
            stock = yf.Ticker(ticker)
            df = stock.history(period=DATA_PERIOD, auto_adjust=True)
            if not df.empty:
                print(f"Загружено {len(df)} записей через Ticker")
        
        # Проверка результата
        if df is None or df.empty:
            return None, f"Не удалось загрузить данные для тикера {ticker}. Возможные причины:\n" \
                        f"- Неверный тикер\n" \
                        f"- Проблемы с соединением\n" \
                        f"- Yahoo Finance временно недоступен\n\n" \
                        f"Попробуйте позже или проверьте тикер на https://finance.yahoo.com"
        
        # Обработка данных
        if 'Close' in df.columns:
            df = df[['Close']].copy()
        elif 'close' in df.columns:
            df = df[['close']].copy()
            df.columns = ['Close']
        else:
            return None, f"Неожиданный формат данных для {ticker}"
        
        df.reset_index(inplace=True)
        
        # Обработка названия колонки с датой
        if 'Date' in df.columns:
            df.columns = ['Date', 'Price']
        elif 'date' in df.columns:
            df.columns = ['Date', 'Price']
        else:
            df.columns = ['Date', 'Price']
        
        # Проверяем достаточность данных
        if len(df) < 100:
            return None, f"Недостаточно исторических данных для тикера {ticker}. " \
                        f"Доступно только {len(df)} записей (нужно минимум 100)."
        
        print(f"Успешно загружено {len(df)} записей для {ticker}")
        return df, None
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Ошибка при загрузке {ticker}: {error_details}")
        return None, f"Ошибка при загрузке данных для {ticker}: {str(e)}\n\n" \
                    f"Попробуйте:\n" \
                    f"- Проверить интернет-соединение\n" \
                    f"- Ввести другой тикер\n" \
                    f"- Повторить попытку позже"


def get_current_price(df: pd.DataFrame) -> float:
    """Возвращает последнюю известную цену акции."""
    return df['Price'].iloc[-1]


def get_company_info(ticker: str) -> dict:
    """Получает информацию о компании."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'currency': info.get('currency', 'USD')
        }
    except:
        return {'name': ticker, 'currency': 'USD'}
