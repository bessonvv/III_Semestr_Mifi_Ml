import numpy as np
from typing import List, Tuple, Dict, Any


def find_local_extrema(forecast: np.ndarray) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Находит локальные минимумы и максимумы в прогнозе.
    
    Args:
        forecast: Массив прогнозных значений
    
    Returns:
        (buy_points, sell_points) - списки кортежей (день, цена) для покупки и продажи
    """
    # Находим локальные минимумы (дни покупки)
    local_minima = []
    for i in range(1, len(forecast) - 1):
        if forecast[i] < forecast[i-1] and forecast[i] < forecast[i+1]:
            local_minima.append((i + 1, forecast[i]))  # i+1 потому что день 1 = завтра
    
    # Находим локальные максимумы (дни продажи)
    local_maxima = []
    for i in range(1, len(forecast) - 1):
        if forecast[i] > forecast[i-1] and forecast[i] > forecast[i+1]:
            local_maxima.append((i + 1, forecast[i]))  # день и цена
    
    return local_minima, local_maxima


def calculate_investment_strategy(
    current_price: float,
    forecast: np.ndarray,
    investment_amount: float
) -> Dict[str, Any]:
    """
    Анализирует прогноз и рассчитывает инвестиционную стратегию.
    
    Args:
        current_price: Текущая цена акции
        forecast: Массив прогнозных значений
        investment_amount: Сумма для инвестиции
    
    Returns:
        Словарь с рекомендациями и расчётами прибыли
    """
    # Находим локальные экстремумы
    buy_points, sell_points = find_local_extrema(forecast)
    
    # Если нет точек для торговли
    if not buy_points and not sell_points:
        return {
            'recommendations': [],
            'total_profit': 0,
            'profit_percentage': 0,
            'final_amount': investment_amount,
            'trades_count': 0
        }
    
    # Создаём торговую стратегию
    trades = []
    cash = investment_amount
    shares = 0
    
    # Объединяем и сортируем все точки
    all_events = []
    for day, price in buy_points:
        all_events.append((day, price, 'buy'))
    for day, price in sell_points:
        all_events.append((day, price, 'sell'))
    
    all_events.sort(key=lambda x: x[0])
    
    # Симулируем торговлю
    for day, price, action in all_events:
        if action == 'buy' and shares == 0:  # Покупаем, если нет позиции
            shares = cash / price
            trades.append({
                'day': day,
                'action': 'Купить',
                'price': price,
                'shares': shares
            })
            cash = 0
        elif action == 'sell' and shares > 0:  # Продаём, если есть позиция
            cash = shares * price
            trades.append({
                'day': day,
                'action': 'Продать',
                'price': price,
                'shares': shares
            })
            shares = 0
    
    # Если остались акции, продаём по последней цене прогноза
    if shares > 0:
        final_price = forecast[-1]
        cash = shares * final_price
        trades.append({
            'day': len(forecast),
            'action': 'Продать (конец периода)',
            'price': final_price,
            'shares': shares
        })
        shares = 0
    
    # Расчёт прибыли
    final_amount = cash if shares == 0 else shares * forecast[-1]
    total_profit = final_amount - investment_amount
    profit_percentage = (total_profit / investment_amount) * 100
    
    return {
        'recommendations': trades,
        'total_profit': total_profit,
        'profit_percentage': profit_percentage,
        'final_amount': final_amount,
        'trades_count': len(trades),
        'buy_points': buy_points,
        'sell_points': sell_points
    }


def format_strategy_message(strategy: Dict[str, Any], investment_amount: float) -> str:
    """
    Форматирует сообщение с торговыми рекомендациями.
    
    Args:
        strategy: Результат расчёта стратегии
        investment_amount: Исходная сумма инвестиции
    
    Returns:
        Отформатированное сообщение
    """
    message = "\n💼 Инвестиционная стратегия:\n\n"
    message += f"Начальная сумма: ${investment_amount:.2f}\n\n"
    
    if strategy['trades_count'] == 0:
        message += "⚠️ Не найдено благоприятных точек для торговли.\n"
        message += "Рекомендация: Держать текущую позицию или дождаться более чётких сигналов.\n"
    else:
        message += "📈 Рекомендации по сделкам:\n\n"
        for i, trade in enumerate(strategy['recommendations'], 1):
            message += f"{i}. День {trade['day']}: {trade['action']} по ${trade['price']:.2f}\n"
        
        message += f"\n💰 Результаты стратегии:\n"
        message += f"Итоговая сумма: ${strategy['final_amount']:.2f}\n"
        message += f"Прибыль: ${strategy['total_profit']:.2f}\n"
        message += f"Доходность: {strategy['profit_percentage']:.2f}%\n"
        message += f"Количество сделок: {strategy['trades_count']}\n"
    
    return message
