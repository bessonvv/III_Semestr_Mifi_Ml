import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class MLModel:
    """Классическая ML-модель на основе Random Forest с лаговыми признаками."""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.name = "Random Forest"
        
    def create_features(self, df: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
        """
        Создаёт лаговые признаки и технические индикаторы.
        
        Args:
            df: DataFrame с ценами
            lags: Количество лагов
        """
        data = df.copy()
        
        # Лаговые признаки
        for i in range(1, lags + 1):
            data[f'lag_{i}'] = data['Price'].shift(i)
        
        # Скользящие средние
        data['ma_7'] = data['Price'].rolling(window=7).mean()
        data['ma_30'] = data['Price'].rolling(window=30).mean()
        
        # Стандартное отклонение
        data['std_7'] = data['Price'].rolling(window=7).std()
        
        # Momentum
        data['momentum'] = data['Price'] - data['Price'].shift(5)
        
        # Rate of change
        data['roc'] = (data['Price'] - data['Price'].shift(5)) / data['Price'].shift(5) * 100
        
        # Удаляем строки с NaN
        data.dropna(inplace=True)
        
        return data
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обучает модель и возвращает предсказания на тестовой выборке.
        
        Returns:
            y_test, y_pred - истинные значения и предсказания
        """
        # Создание признаков
        data = self.create_features(df)
        
        # Разделение на признаки и целевую переменную
        X = data.drop(['Date', 'Price'], axis=1)
        y = data['Price'].values
        
        # Разделение на train/test
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Масштабирование
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Обучение
        self.model.fit(X_train_scaled, y_train)
        
        # Предсказание на тестовой выборке
        y_pred = self.model.predict(X_test_scaled)
        
        return y_test, y_pred
    
    def predict(self, df: pd.DataFrame, days: int = 30) -> np.ndarray:
        """
        Прогнозирует цены на указанное количество дней вперёд.
        
        Args:
            df: DataFrame с историческими данными
            days: Количество дней для прогноза
        """
        data = self.create_features(df)
        predictions = []
        
        # Последние известные данные
        last_data = data.drop(['Date', 'Price'], axis=1).iloc[-1:].copy()
        current_price = df['Price'].iloc[-1]
        
        for _ in range(days):
            # Предсказание следующего значения
            X_scaled = self.scaler.transform(last_data)
            next_price = self.model.predict(X_scaled)[0]
            predictions.append(next_price)
            
            # Обновление признаков для следующей итерации
            # Сдвигаем лаги
            for i in range(10, 1, -1):
                last_data[f'lag_{i}'] = last_data[f'lag_{i-1}'].values
            last_data['lag_1'] = next_price
            
            # Обновляем скользящие средние (упрощённо)
            current_price = next_price
        
        return np.array(predictions)
