# Простые числа на Python

___

### Запуск GPU OpenCL (AMD) версии решета

1. Установить Anaconda
2. Используйте Anaconda Prompt && cd на проект
3. `conda install -c conda-forge pyopencl`
4. `conda create --name myenv python=3.9`
5. `conda activate myenv`
6. `conda install -c conda-forge numpy pandas matplotlib scikit-learn plotly datashader holoviews bokeh`
7. `python .\Eratosthenes_OpenCL.py`

### Проверка

1. `conda install -c conda-forge clinfo`

### Визуализация

1. `python ./visual_png.py`