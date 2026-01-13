import numpy as np
import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Подавляем TensorFlow предупреждения
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


class LSTMModel:
    """Нейросетевая модель на основе LSTM для прогнозирования временных рядов."""
    
    def __init__(self, lookback: int = 60):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lookback = lookback
        self.name = "LSTM"
        
    def prepare_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготавливает данные для LSTM: создаёт последовательности.
        
        Args:
            data: Массив цен
        
        Returns:
            X, y - входные последовательности и целевые значения
        """
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, 0])
            y.append(data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape для LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self, input_shape: tuple) -> Sequential:
        """Создаёт архитектуру LSTM."""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обучает LSTM модель и возвращает предсказания на тестовой выборке.
        
        Returns:
            y_test, y_pred - истинные значения и предсказания
        """
        # Нормализация данных
        prices = df['Price'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(prices)
        
        # Подготовка последовательностей
        X, y = self.prepare_data(scaled_data)
        
        # Разделение на train/test
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Построение и обучение модели
        self.model = self.build_model((X_train.shape[1], 1))
        
        early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0)
        
        self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Предсказание на тестовой выборке
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        
        # Денормализация
        y_pred = self.scaler.inverse_transform(y_pred_scaled).flatten()
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        return y_test_original, y_pred
    
    def predict(self, df: pd.DataFrame, days: int = 30) -> np.ndarray:
        """
        Прогнозирует цены на указанное количество дней вперёд.
        
        Args:
            df: DataFrame с историческими данными
            days: Количество дней для прогноза
        """
        # Нормализация данных
        prices = df['Price'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(prices)
        
        # Подготовка последовательностей для обучения
        X, y = self.prepare_data(scaled_data)
        
        # Обучение на всех данных
        self.model = self.build_model((X.shape[1], 1))
        
        early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0)
        
        self.model.fit(
            X, y,
            batch_size=32,
            epochs=50,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Прогнозирование
        predictions = []
        last_sequence = scaled_data[-self.lookback:].copy()
        
        for _ in range(days):
            # Подготовка входных данных
            X_pred = last_sequence.reshape(1, self.lookback, 1)
            
            # Предсказание следующего значения
            next_pred = self.model.predict(X_pred, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Обновление последовательности
            last_sequence = np.append(last_sequence[1:], next_pred)
            last_sequence = last_sequence.reshape(-1, 1)
        
        # Денормализация предсказаний
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions
