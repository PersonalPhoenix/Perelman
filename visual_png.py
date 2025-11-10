import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import glob
import os
import time
from datashader import Canvas
import datashader.transfer_functions as tf
from datashader.colors import viridis
import holoviews as hv
from holoviews.operation.datashader import datashade
hv.extension('bokeh')

def load_primes_data():
    """Загрузка всех простых чисел из CSV файлов"""
    print("Загрузка данных...")
    csv_files = glob.glob('*.csv')
    all_primes = []
    
    for file in csv_files:
        try:
            # Читаем CSV, предполагая что простые числа в первом столбце
            df = pd.read_csv(file, header=None, names=['prime'])
            all_primes.extend(df['prime'].values)
            print(f"Загружено {len(df)} простых чисел из {file}")
        except Exception as e:
            print(f"Ошибка чтения {file}: {e}")
    
    all_primes = np.array(all_primes, dtype=np.int64)
    all_primes.sort()
    print(f"Всего загружено простых чисел: {len(all_primes)}")
    return all_primes

def create_visualization_data(primes):
    """Создание данных для визуализации"""
    print("Подготовка данных для визуализации...")
    
    # Создаем последовательные индексы (номера простых чисел)
    indices = np.arange(1, len(primes) + 1)
    
    # Для логарифмического масштаба
    log_indices = np.log10(indices)
    log_primes = np.log10(primes)
    
    return {
        'linear': {'x': indices, 'y': primes},
        'log_log': {'x': log_indices, 'y': log_primes},
        'log_linear': {'x': indices, 'y': primes}  # для полулогарифмического
    }

def calculate_trend_line(data):
    """Вычисление линии тренда (линейной регрессии)"""
    print("Вычисление линии тренда...")
    
    # Для логарифмических данных
    X_log = data['log_log']['x'].reshape(-1, 1)
    y_log = data['log_log']['y']
    
    model = LinearRegression()
    model.fit(X_log, y_log)
    
    trend_line_log = model.predict(X_log)
    
    # Преобразуем обратно в линейные координаты для отображения
    trend_line_linear = 10**trend_line_log
    x_linear = 10**data['log_log']['x']
    
    return {
        'log_trend': trend_line_log,
        'linear_trend': trend_line_linear,
        'linear_x': x_linear,
        'r_squared': model.score(X_log, y_log)
    }

def plot_with_datashader(data, trend_data):
    """Визуализация с использованием Datashader для больших данных"""
    print("Создание визуализации с Datashader...")
    
    # Создаем DataFrame для Datashader
    df = pd.DataFrame({
        'index': data['linear']['x'],
        'prime': data['linear']['y'],
        'log_index': data['log_log']['x'],
        'log_prime': data['log_log']['y']
    })
    
    # Создаем canvas для линейного масштаба
    canvas_linear = Canvas(x_range=(df['index'].min(), df['index'].max()),
                          y_range=(df['prime'].min(), df['prime'].max()),
                          plot_width=800, plot_height=600)
    
    # Аггрегируем точки
    agg_linear = canvas_linear.points(df, 'index', 'prime')
    img_linear = tf.shade(agg_linear, cmap=viridis)
    
    # Создаем canvas для логарифмического масштаба
    canvas_log = Canvas(x_range=(df['log_index'].min(), df['log_index'].max()),
                       y_range=(df['log_prime'].min(), df['log_prime'].max()),
                       plot_width=800, plot_height=600)
    
    agg_log = canvas_log.points(df, 'log_index', 'log_prime')
    img_log = tf.shade(agg_log, cmap=viridis)
    
    return img_linear, img_log

def create_matplotlib_plots(data, trend_data):
    """Создание графиков с использованием matplotlib"""
    print("Создание графиков matplotlib...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Линейный масштаб (с выборкой для производительности)
    sample_size = min(100000, len(data['linear']['x']))
    step = len(data['linear']['x']) // sample_size
    x_linear_sample = data['linear']['x'][::step]
    y_linear_sample = data['linear']['y'][::step]
    
    ax1.scatter(x_linear_sample, y_linear_sample, s=0.1, alpha=0.5, color='blue')
    ax1.plot(trend_data['linear_x'], trend_data['linear_trend'], 'r-', linewidth=2, 
             label=f'Тренд (R²={trend_data["r_squared"]:.4f})')
    ax1.set_xlabel('Номер простого числа')
    ax1.set_ylabel('Простое число')
    ax1.set_title('Линейный масштаб\nРаспределение простых чисел')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Логарифмический масштаб (логарифм по обеим осям)
    ax2.scatter(data['log_log']['x'], data['log_log']['y'], s=0.1, alpha=0.5, color='green')
    ax2.plot(data['log_log']['x'], trend_data['log_trend'], 'r-', linewidth=2,
             label=f'Линейный тренд (R²={trend_data["r_squared"]:.4f})')
    ax2.set_xlabel('log₁₀(Номер простого числа)')
    ax2.set_ylabel('log₁₀(Простое число)')
    ax2.set_title('Логарифмический масштаб (log-log)\nЛинейная зависимость в логарифмических координатах')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Полулогарифмический масштаб (только по Y)
    ax3.scatter(x_linear_sample, y_linear_sample, s=0.1, alpha=0.5, color='purple')
    ax3.set_yscale('log')
    ax3.set_xlabel('Номер простого числа')
    ax3.set_ylabel('Простое число (лог. шкала)')
    ax3.set_title('Полулогарифмический масштаб')
    ax3.grid(True, alpha=0.3)
    
    # Гистограмма распределения (логарифмическая шкала)
    ax4.hist(data['linear']['y'], bins=100, alpha=0.7, color='orange', log=True)
    ax4.set_xlabel('Простое число')
    ax4.set_ylabel('Частота (лог. шкала)')
    ax4.set_title('Гистограмма распределения простых чисел')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    start_time = time.time()
    
    # Загружаем данные
    primes = load_primes_data()
    
    if len(primes) == 0:
        print("Не найдено простых чисел для визуализации")
        return
    
    # Подготавливаем данные
    data = create_visualization_data(primes)
    
    # Вычисляем тренд
    trend_data = calculate_trend_line(data)
    
    print(f"Качество аппроксимации (R²): {trend_data['r_squared']:.4f}")
    
    # Создаем графики matplotlib
    fig = create_matplotlib_plots(data, trend_data)
    
    # Сохраняем графики
    plt.savefig('primes_visualization.png', dpi=300, bbox_inches='tight')
    print("График сохранен как 'primes_visualization.png'")
    
    # Дополнительная статистика
    print("\nСтатистика:")
    print(f"Всего простых чисел: {len(primes):,}")
    print(f"Минимальное: {primes[0]:,}")
    print(f"Максимальное: {primes[-1]:,}")
    print(f"Медиана: {np.median(primes):,}")
    print(f"Среднее: {np.mean(primes):,.2f}")
    
    total_time = time.time() - start_time
    print(f"\nВремя выполнения: {total_time:.2f} секунд")
    
    # Показываем графики
    plt.show()

if __name__ == "__main__":
    main()