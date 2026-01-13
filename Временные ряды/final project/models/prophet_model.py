import numpy as np
import pandas as pd
from prophet import Prophet
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)


class ProphetModel:
    """Статистическая модель прогнозирования на основе Prophet."""
    
    def __init__(self):
        self.model = None
        self.name = "Prophet"
        
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обучает модель Prophet и возвращает предсказания на тестовой выборке.
        
        Returns:
            y_test, y_pred - истинные значения и предсказания
        """
        # Подготовка данных для Prophet
        prophet_df = df[['Date', 'Price']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Разделение на train/test
        split_idx = int(len(prophet_df) * (1 - test_size))
        train_df = prophet_df[:split_idx]
        test_df = prophet_df[split_idx:]
        
        # Обучение модели с подавлением вывода
        self.model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative'
        )
        
        # Подавляем вывод Prophet
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            self.model.fit(train_df)
        finally:
            sys.stdout = old_stdout
        
        # Предсказание на тестовой выборке
        future = test_df[['ds']].copy()
        forecast = self.model.predict(future)
        
        y_test = test_df['y'].values
        y_pred = forecast['yhat'].values
        
        return y_test, y_pred
    
    def predict(self, df: pd.DataFrame, days: int = 30) -> np.ndarray:
        """
        Прогнозирует цены на указанное количество дней вперёд.
        
        Args:
            df: DataFrame с историческими данными
            days: Количество дней для прогноза
        """
        # Подготовка данных
        prophet_df = df[['Date', 'Price']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Обучение на всех данных с подавлением вывода
        self.model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative'
        )
        
        # Подавляем вывод Prophet
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            self.model.fit(prophet_df)
        finally:
            sys.stdout = old_stdout
        
        # Создание будущих дат
        future = self.model.make_future_dataframe(periods=days)
        
        # Прогнозирование
        forecast = self.model.predict(future)
        
        # Возвращаем только прогноз (последние days значений)
        predictions = forecast['yhat'].tail(days).values
        
        return predictions
