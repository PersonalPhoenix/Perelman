import math
import csv
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def worker_calculate_primes(range_start, range_end, filename, worker_id):
    """Рабочая функция для обработки своего диапазона чисел"""
    primes_found = []
    start_time = time.time()
    last_report_time = start_time
    
    print(f"Рабочий {worker_id}: начал диапазон {range_start:,} - {range_end:,}")
    
    for n in range(range_start, range_end + 1):
        if is_prime(n):
            primes_found.append(n)
            
            # Периодическая отчетность о прогрессе
            current_time = time.time()
            if current_time - last_report_time > 5:  # Каждые 5 секунд
                progress = (n - range_start) / (range_end - range_start) * 100
                primes_per_sec = len(primes_found) / (current_time - start_time)
                print(f"Рабочий {worker_id}: {progress:.1f}% | "
                      f"Простых: {len(primes_found)} | "
                      f"Скорость: {primes_per_sec:.2f} простых/сек | "
                      f"Текущее: {n:,}")
                last_report_time = current_time
    
    # Записываем результаты в файл
    if primes_found:
        with open(f"{filename}_worker_{worker_id}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            for prime in primes_found:
                writer.writerow([prime])
    
    total_time = time.time() - start_time
    print(f"Рабочий {worker_id} завершил за {total_time:.1f}с. "
          f"Найдено простых: {len(primes_found)}")
    
    return {
        'worker_id': worker_id,
        'primes_found': len(primes_found),
        'range_processed': (range_start, range_end),
        'total_time': total_time,
        'primes': primes_found
    }

def merge_results(main_filename, num_workers):
    """Объединение результатов из временных файлов в основной файл"""
    all_primes = []
    
    # Собираем все простые числа из временных файлов
    for i in range(num_workers):
        temp_filename = f"{main_filename}_worker_{i}.csv"
        if os.path.exists(temp_filename):
            with open(temp_filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        all_primes.append(int(row[0]))
            # Удаляем временный файл
            os.remove(temp_filename)
    
    # Сортируем и записываем в основной файл
    all_primes.sort()
    with open(main_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for prime in all_primes:
            writer.writerow([prime])
    
    return len(all_primes)

def create_ranges_power_based(num_workers, start_power=10, power_step=5):
    """Создание диапазонов на основе степеней 10"""
    ranges = []
    current_start = 10 ** start_power
    
    for i in range(num_workers):
        range_end = current_start * (10 ** power_step) - 1
        ranges.append((current_start, range_end))
        current_start = range_end + 1
    
    return ranges

def create_ranges_linear(num_workers, start_num, total_range_size):
    """Создание линейных диапазонов равного размера"""
    ranges = []
    range_size = total_range_size // num_workers
    
    for i in range(num_workers):
        range_start = start_num + i * range_size
        range_end = range_start + range_size - 1
        if i == num_workers - 1:  # Последний рабочий получает остаток
            range_end = start_num + total_range_size - 1
        ranges.append((range_start, range_end))
    
    return ranges

class DistributedPrimeCalculator:
    def __init__(self, filename='primes.csv'):
        self.filename = filename
        self.start_num = 2
        
        # Чтение существующих данных
        self._read_existing_primes()
    
    def _read_existing_primes(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                reader = csv.reader(f)
                primes = []
                for row in reader:
                    if row:
                        try:
                            prime = int(row[0])
                            primes.append(prime)
                        except ValueError:
                            pass
                if primes:
                    self.start_num = primes[-1] + 1
                    print(f"Продолжаем с числа {self.start_num:,}, найдено {len(primes)} простых чисел")
    
    def calculate_distributed(self, strategy='power', **kwargs):
        num_workers = mp.cpu_count()
        print(f"Запуск распределенного вычисления на {num_workers} ядрах")
        print(f"Стратегия распределения: {strategy}")
        
        program_start_time = time.time()
        
        # Создаем диапазоны в зависимости от стратегии
        if strategy == 'power':
            start_power = kwargs.get('start_power', 10)
            power_step = kwargs.get('power_step', 5)
            ranges = create_ranges_power_based(num_workers, start_power, power_step)
        else:  # linear
            total_range_size = kwargs.get('total_range_size', 10**9)
            ranges = create_ranges_linear(num_workers, self.start_num, total_range_size)
        
        # Выводим информацию о диапазонах
        print("\nРаспределение диапазонов:")
        for i, (start, end) in enumerate(ranges):
            print(f"  Рабочий {i}: {start:,} - {end:,} "
                  f"(размер: {end - start + 1:,} чисел)")
        
        # Запускаем рабочих процессов
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i, (range_start, range_end) in enumerate(ranges):
                future = executor.submit(
                    worker_calculate_primes, 
                    range_start, range_end, self.filename, i
                )
                futures.append(future)
            
            # Собираем результаты
            completed_workers = 0
            for future in as_completed(futures):
                result = future.result()
                completed_workers += 1
                print(f"\nЗавершено workers: {completed_workers}/{num_workers}")
                print(f"  Рабочий {result['worker_id']}: "
                      f"{result['primes_found']} простых чисел, "
                      f"время: {result['total_time']:.1f}с")
        
        # Объединяем результаты
        print("\nОбъединение результатов...")
        total_primes = merge_results(self.filename, num_workers)
        
        total_time = time.time() - program_start_time
        print(f"\n=== ВЫПОЛНЕНИЕ ЗАВЕРШЕНО ===")
        print(f"Общее время: {total_time:.1f} секунд")
        print(f"Всего найдено простых чисел: {total_primes}")
        print(f"Скорость: {total_primes/total_time:.2f} простых/сек")

def main():
    calculator = DistributedPrimeCalculator()
    
    print("Выберите стратегию распределения:")
    print("1. По степеням 10 (для очень больших чисел)")
    print("2. Линейное распределение (для последовательного диапазона)")
    
    choice = input("Введите 1 или 2: ").strip()
    
    if choice == "1":
        start_power = int(input("Начальная степень 10 (например, 10 для 10^10): ") or "10")
        power_step = int(input("Шаг степени для каждого worker (например, 5): ") or "5")
        calculator.calculate_distributed(
            strategy='power',
            start_power=start_power,
            power_step=power_step
        )
    else:
        total_range = int(input("Общий размер диапазона (например, 1000000000): ") or "1000000000")
        calculator.calculate_distributed(
            strategy='linear',
            total_range_size=total_range
        )

if __name__ == "__main__":
    mp.freeze_support()
    main()