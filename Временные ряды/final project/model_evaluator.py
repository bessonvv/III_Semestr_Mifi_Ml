import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from models import MLModel, ARIMAModel, LSTMModel
from config import TEST_SIZE, FORECAST_DAYS


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Вычисляет метрики качества прогнозирования.
    
    Args:
        y_true: Истинные значения
        y_pred: Предсказанные значения
    
    Returns:
        Словарь с метриками
    """
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }


def evaluate_models(df: pd.DataFrame) -> Tuple[Any, str, Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Обучает все модели, сравнивает их и выбирает лучшую.
    
    Args:
        df: DataFrame с историческими данными
    
    Returns:
        best_model, best_model_name, best_metrics, all_metrics
    """
    models = [
        MLModel(),
        ARIMAModel(),
        LSTMModel()
    ]
    
    results = {}
    all_metrics = {}
    
    print("Обучение и оценка моделей...")
    
    for model in models:
        try:
            print(f"  Обучение {model.name}...")
            y_test, y_pred = model.train(df, test_size=TEST_SIZE)
            metrics = calculate_metrics(y_test, y_pred)
            
            results[model.name] = {
                'model': model,
                'metrics': metrics
            }
            all_metrics[model.name] = metrics
            
            print(f"    {model.name} - RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")
            
        except Exception as e:
            print(f"  Ошибка при обучении {model.name}: {e}")
            continue
    
    if not results:
        raise ValueError("Не удалось обучить ни одну модель")
    
    # Выбор лучшей модели по RMSE (можно изменить критерий)
    best_model_name = min(results.keys(), key=lambda x: results[x]['metrics']['RMSE'])
    best_model = results[best_model_name]['model']
    best_metrics = results[best_model_name]['metrics']
    
    print(f"\nЛучшая модель: {best_model_name}")
    
    return best_model, best_model_name, best_metrics, all_metrics


def make_forecast(model: Any, df: pd.DataFrame, days: int = FORECAST_DAYS) -> np.ndarray:
    """
    Делает прогноз на указанное количество дней.
    
    Args:
        model: Обученная модель
        df: DataFrame с историческими данными
        days: Количество дней для прогноза
    
    Returns:
        Массив с прогнозными значениями
    """
    print(f"Создание прогноза на {days} дней...")
    forecast = model.predict(df, days=days)
    return forecast
