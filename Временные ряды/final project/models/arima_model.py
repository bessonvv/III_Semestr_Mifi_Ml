import numpy as np
import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAModel:
    """Статистическая модель ARIMA для прогнозирования временных рядов."""
    
    def __init__(self, order=(5, 1, 0)):
        self.model = None
        self.order = order
        self.name = "ARIMA"
        self.fitted_model = None
        
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обучает модель ARIMA и возвращает предсказания на тестовой выборке.
        
        Returns:
            y_test, y_pred - истинные значения и предсказания
        """
        # Подготовка данных
        prices = df['Price'].values
        
        # Разделение на train/test
        split_idx = int(len(prices) * (1 - test_size))
        train_data = prices[:split_idx]
        test_data = prices[split_idx:]
        
        try:
            # Обучение ARIMA модели
            self.model = ARIMA(train_data, order=self.order)
            self.fitted_model = self.model.fit()
            
            # Прогнозирование на тестовой выборке
            forecast = self.fitted_model.forecast(steps=len(test_data))
            
            return test_data, forecast
            
        except Exception as e:
            # Если не получается с заданными параметрами, пробуем более простую модель
            print(f"    Попытка с упрощёнными параметрами...")
            try:
                self.order = (1, 1, 1)
                self.model = ARIMA(train_data, order=self.order)
                self.fitted_model = self.model.fit()
                forecast = self.fitted_model.forecast(steps=len(test_data))
                return test_data, forecast
            except Exception as e2:
                raise ValueError(f"Не удалось обучить ARIMA: {e2}")
    
    def predict(self, df: pd.DataFrame, days: int = 30) -> np.ndarray:
        """
        Прогнозирует цены на указанное количество дней вперёд.
        
        Args:
            df: DataFrame с историческими данными
            days: Количество дней для прогноза
        """
        prices = df['Price'].values
        
        try:
            # Обучение на всех данных
            self.model = ARIMA(prices, order=self.order)
            self.fitted_model = self.model.fit()
            
            # Прогнозирование
            forecast = self.fitted_model.forecast(steps=days)
            
            return forecast
            
        except Exception as e:
            # Если не получается, пробуем упрощённую модель
            print(f"    Использую упрощённую ARIMA модель...")
            try:
                self.order = (1, 1, 1)
                self.model = ARIMA(prices, order=self.order)
                self.fitted_model = self.model.fit()
                forecast = self.fitted_model.forecast(steps=days)
                return forecast
            except Exception as e2:
                # В крайнем случае используем простое экспоненциальное сглаживание
                print(f"    Использую простое сглаживание...")
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                model = ExponentialSmoothing(prices, trend='add', seasonal=None)
                fitted = model.fit()
                forecast = fitted.forecast(steps=days)
                return forecast
