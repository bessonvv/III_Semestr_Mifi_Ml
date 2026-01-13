import asyncio
import os
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import FSInputFile

from config import BOT_TOKEN
from data_loader import load_stock_data, get_current_price, get_company_info
from model_evaluator import evaluate_models, make_forecast
from visualization import create_forecast_plot, format_price_change
from trading_strategy import calculate_investment_strategy, format_strategy_message
from logger import log_request, create_log_header_if_needed


# Состояния для FSM
class ForecastStates(StatesGroup):
    waiting_for_ticker = State()
    waiting_for_amount = State()


# Инициализация бота
bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)


@dp.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    """Обработчик команды /start"""
    await state.clear()
    
    welcome_text = """
👋 Добро пожаловать в бота прогнозирования акций!

Я помогу вам:
📈 Получить прогноз цены акций на 30 дней
🤖 Выбрать лучшую модель машинного обучения
💼 Рассчитать потенциальную инвестиционную стратегию
📊 Визуализировать результаты

Для начала введите тикер компании (например: AAPL, MSFT, GOOGL, TSLA)
"""
    
    await message.answer(welcome_text)
    await state.set_state(ForecastStates.waiting_for_ticker)


@dp.message(ForecastStates.waiting_for_ticker)
async def process_ticker(message: types.Message, state: FSMContext):
    """Обработка ввода тикера"""
    ticker = message.text.strip().upper()
    
    # Проверка валидности тикера (базовая)
    if not ticker.isalpha() or len(ticker) > 5:
        await message.answer("❌ Некорректный тикер. Попробуйте снова (например: AAPL, MSFT)")
        return
    
    await state.update_data(ticker=ticker)
    await message.answer(
        f"✅ Тикер: {ticker}\n\n"
        "Теперь введите сумму для условной инвестиции (в долларах, например: 10000):"
    )
    await state.set_state(ForecastStates.waiting_for_amount)


@dp.message(ForecastStates.waiting_for_amount)
async def process_amount(message: types.Message, state: FSMContext):
    """Обработка ввода суммы инвестиции"""
    try:
        amount = float(message.text.strip().replace(',', ''))
        
        if amount <= 0:
            await message.answer("❌ Сумма должна быть положительной. Попробуйте снова:")
            return
        
        if amount > 1000000000:
            await message.answer("❌ Слишком большая сумма. Попробуйте снова:")
            return
        
    except ValueError:
        await message.answer("❌ Некорректная сумма. Введите число (например: 10000):")
        return
    
    # Получаем сохранённые данные
    data = await state.get_data()
    ticker = data['ticker']
    
    # Начинаем обработку
    processing_msg = await message.answer(
        f"⏳ Обрабатываю запрос...\n\n"
        f"Тикер: {ticker}\n"
        f"Сумма инвестиции: ${amount:,.2f}\n\n"
        f"Это может занять несколько минут. Пожалуйста, подождите..."
    )
    
    try:
        # 1. Загрузка данных
        await processing_msg.edit_text(
            f"{processing_msg.text}\n\n"
            f"📥 Загружаю исторические данные..."
        )
        
        df, error = load_stock_data(ticker)
        if error:
            await processing_msg.edit_text(f"❌ {error}")
            await state.clear()
            return
        
        company_info = get_company_info(ticker)
        current_price = get_current_price(df)
        
        # 2. Обучение моделей
        await processing_msg.edit_text(
            f"{processing_msg.text}\n"
            f"✅ Данные загружены\n\n"
            f"🤖 Обучаю модели машинного обучения..."
        )
        
        best_model, best_model_name, best_metrics, all_metrics = evaluate_models(df)
        
        # 3. Прогнозирование
        await processing_msg.edit_text(
            f"{processing_msg.text}\n"
            f"✅ Лучшая модель: {best_model_name}\n\n"
            f"🔮 Создаю прогноз..."
        )
        
        forecast = make_forecast(best_model, df)
        
        # 4. Визуализация
        await processing_msg.edit_text(
            f"{processing_msg.text}\n"
            f"✅ Прогноз создан\n\n"
            f"📊 Генерирую график..."
        )
        
        plot_path = create_forecast_plot(df, forecast, ticker, f'forecast_{ticker}.png')
        
        # 5. Расчёт стратегии
        await processing_msg.edit_text(
            f"{processing_msg.text}\n"
            f"✅ График готов\n\n"
            f"💼 Рассчитываю стратегию..."
        )
        
        strategy = calculate_investment_strategy(current_price, forecast, amount)
        
        # 6. Логирование
        log_request(
            user_id=message.from_user.id,
            ticker=ticker,
            investment_amount=amount,
            model_name=best_model_name,
            metric_value=best_metrics['RMSE'],
            profit=strategy['total_profit']
        )
        
        # 7. Отправка результатов
        await processing_msg.edit_text(
            f"{processing_msg.text}\n"
            f"✅ Готово!\n\n"
            f"📤 Отправляю результаты..."
        )
        
        # Информация о компании и модели
        info_message = f"""
📊 Результаты анализа для {company_info['name']} ({ticker})

🤖 Использованная модель: {best_model_name}
📉 Метрики качества:
  • RMSE: {best_metrics['RMSE']:.2f}
  • MAPE: {best_metrics['MAPE']:.2f}%
  • MAE: {best_metrics['MAE']:.2f}
"""
        
        await message.answer(info_message)
        
        # График
        photo = FSInputFile(plot_path)
        await message.answer_photo(photo)
        
        # Прогноз изменения цены
        price_change_msg = format_price_change(current_price, forecast)
        await message.answer(price_change_msg)
        
        # Торговая стратегия
        strategy_msg = format_strategy_message(strategy, amount)
        await message.answer(strategy_msg)
        
        # Удаляем временный файл графика
        if os.path.exists(plot_path):
            os.remove(plot_path)
        
        # Предложение нового анализа
        await message.answer(
            "\n✨ Хотите проанализировать другую акцию?\n"
            "Введите новый тикер или отправьте /start для начала."
        )
        
        await state.set_state(ForecastStates.waiting_for_ticker)
        
    except Exception as e:
        error_message = f"❌ Произошла ошибка при обработке: {str(e)}\n\nПопробуйте снова с помощью /start"
        await message.answer(error_message)
        await state.clear()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    """Обработчик команды /help"""
    help_text = """
📖 Инструкция по использованию:

1️⃣ Отправьте /start для начала
2️⃣ Введите тикер компании (например: AAPL для Apple)
3️⃣ Укажите сумму для условной инвестиции
4️⃣ Дождитесь результатов анализа

📊 Бот предоставит:
• Прогноз цены на 30 дней
• График с визуализацией
• Торговые рекомендации
• Расчёт потенциальной прибыли

💡 Популярные тикеры:
• AAPL - Apple
• MSFT - Microsoft
• GOOGL - Google
• TSLA - Tesla
• AMZN - Amazon
• NVDA - NVIDIA

⚠️ Важно: Прогнозы носят информационный характер и не являются финансовой консультацией.
"""
    
    await message.answer(help_text)


async def main():
    """Главная функция запуска бота"""
    print("🤖 Бот запускается...")
    create_log_header_if_needed()
    print("📝 Логирование настроено")
    print("✅ Бот запущен и готов к работе!")
    
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
