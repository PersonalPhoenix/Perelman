import pandas as pd
import holoviews as hv
from holoviews.operation.datashader import datashade
hv.extension('bokeh')

# Загрузка данных
def create_interactive_plot():
    csv_files = glob.glob('*.csv')
    primes_list = []
    
    for file in csv_files:
        df = pd.read_csv(file, header=None, names=['prime'])
        primes_list.extend(df['prime'].values)
    
    primes = sorted(primes_list)
    indices = list(range(1, len(primes) + 1))
    
    # Создаем HoloViews объекты
    points = hv.Points((indices, primes), ['Index', 'Prime'])
    
    # Применяем datashader для больших данных
    shaded = datashade(points, width=800, height=600, cmap='viridis')
    
    return shaded

# Сохраняем как HTML для просмотра в браузере
plot = create_interactive_plot()
hv.save(plot, 'interactive_primes.html')