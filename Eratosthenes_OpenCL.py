import pyopencl as cl
import numpy as np
import csv
import os
import time
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import threading

# ОПТИМИЗИРОВАННАЯ КОНФИГУРАЦИЯ
CSV_FILENAME = 'primes.csv'
GPU_BATCH_SIZE = 2000000    # Увеличили размер батча для GPU
CPU_BATCH_SIZE = 50000     # 50K чисел для CPU

def get_last_prime_from_csv(filename):
    if not os.path.exists(filename):
        return 1
    primes = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                try:
                    primes.append(int(row[0]))
                except ValueError:
                    pass
    return primes[-1] if primes else 1

def is_prime_cpu(n):
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

def cpu_prime_worker(args):
    start, end = args
    primes = []
    for n in range(start, end + 1):
        if is_prime_cpu(n):
            primes.append(n)
    return primes

# Оптимизированное ядро OpenCL с векторизацией
opencl_kernel_code = """
__kernel void find_primes(__global const uint* numbers, 
                          __global uint* results,
                          const uint count) {
    uint gid = get_global_id(0);
    if (gid >= count) return;
    
    uint n = numbers[gid];
    
    if (n <= 1) {
        results[gid] = 0;
        return;
    }
    if (n == 2) {
        results[gid] = 1;
        return;
    }
    if (n % 2 == 0) {
        results[gid] = 0;
        return;
    }
    
    // Оптимизация: предварительная проверка маленьких делителей
    if (n > 3 && n % 3 == 0) {
        results[gid] = 0;
        return;
    }
    if (n > 5 && n % 5 == 0) {
        results[gid] = 0;
        return;
    }
    if (n > 7 && n % 7 == 0) {
        results[gid] = 0;
        return;
    }
    
    uint sqrt_n = sqrt((float)n);
    for (uint i = 11; i <= sqrt_n; i += 2) {
        if (n % i == 0) {
            results[gid] = 0;
            return;
        }
    }
    
    results[gid] = 1;
}
"""

class OptimizedPrimeCalculator:
    def __init__(self):
        self.filename = CSV_FILENAME
        self.start_num = get_last_prime_from_csv(self.filename) + 1
        
        # Инициализация ВСЕХ устройств OpenCL
        platforms = cl.get_platforms()
        self.contexts = []
        self.queues = []
        self.programs = []
        self.device_info = []
        
        for platform in platforms:
            try:
                # Пробуем использовать все GPU устройства
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                for device in devices:
                    try:
                        ctx = cl.Context([device])
                        # Создаем несколько очередей для лучшей утилизации
                        queues = [cl.CommandQueue(ctx) for _ in range(2)]
                        program = cl.Program(ctx, opencl_kernel_code).build()
                        
                        self.contexts.append(ctx)
                        self.queues.append(queues)  # Теперь список списков очередей
                        self.programs.append(program)
                        self.device_info.append({
                            'vendor': device.vendor,
                            'name': device.name,
                            'max_work_group_size': device.max_work_group_size,
                            'compute_units': device.max_compute_units
                        })
                        
                        print(f"Успешно инициализировано: {device.vendor} - {device.name}")
                        print(f"  Макс. рабочая группа: {device.max_work_group_size}")
                        print(f"  Выч. единицы: {device.max_compute_units}")
                        print(f"  Очередей: {len(queues)}")
                    except Exception as e:
                        print(f"Ошибка инициализации {device.name}: {e}")
            except:
                continue
        
        if not self.contexts:
            print("Не найдено GPU устройств, используем только CPU")
    
    def calculate_primes_gpu_batch(self, device_idx, start, end):
        """GPU вычисления с использованием нескольких очередей"""
        if device_idx >= len(self.contexts):
            return []
        
        ctx = self.contexts[device_idx]
        queues = self.queues[device_idx]  # Список очередей для этого устройства
        program = self.programs[device_idx]
        device = ctx.devices[0]
        
        numbers = np.arange(start, end + 1, dtype=np.uint32)
        total_numbers = len(numbers)
        
        # Разделяем работу между очередями
        chunk_size = total_numbers // len(queues)
        all_primes = []
        
        def process_chunk(queue, chunk_start, chunk_end):
            chunk_numbers = numbers[chunk_start:chunk_end]
            chunk_total = len(chunk_numbers)
            
            if chunk_total == 0:
                return []
            
            # Оптимизация: используем оптимальный размер рабочей группы
            max_wg_size = device.max_work_group_size
            work_group_size = min(64, max_wg_size)  # Меньше для лучшего распределения
            
            numbers_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=chunk_numbers)
            results_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, chunk_numbers.nbytes)
            
            global_size = ((chunk_total + work_group_size - 1) // work_group_size) * work_group_size
            
            program.find_primes(queue, (global_size,), (work_group_size,), 
                              numbers_buf, results_buf, np.uint32(chunk_total))
            
            results = np.empty(chunk_total, dtype=np.uint32)
            cl.enqueue_copy(queue, results, results_buf).wait()
            
            chunk_primes = chunk_numbers[results == 1]
            return chunk_primes.tolist()
        
        # Запускаем обработку чанков в параллельных потоках
        threads = []
        chunk_results = [[] for _ in range(len(queues))]
        
        for i, queue in enumerate(queues):
            chunk_start = i * chunk_size
            chunk_end = chunk_start + chunk_size if i < len(queues) - 1 else total_numbers
            
            thread = threading.Thread(
                target=lambda idx=i, start=chunk_start, end=chunk_end: 
                chunk_results[idx].extend(process_chunk(queues[idx], start, end))
            )
            threads.append(thread)
            thread.start()
        
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        
        # Собираем результаты
        for result in chunk_results:
            all_primes.extend(result)
        
        print(f"GPU {device_idx} ({self.device_info[device_idx]['name']}): "
              f"{len(all_primes)} primes from {total_numbers} numbers")
        
        return all_primes
    
    def calculate_primes_cpu_parallel(self, start, end):
        num_workers = mp.cpu_count()
        chunk_size = max((end - start) // num_workers, 1)
        
        ranges = []
        current = start
        while current <= end:
            chunk_end = min(current + chunk_size, end)
            ranges.append((current, chunk_end))
            current = chunk_end + 1
        
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(cpu_prime_worker, ranges))
        
        primes = []
        for result in results:
            primes.extend(result)
        
        cpu_time = time.time() - start_time
        total_numbers = end - start + 1
        print(f"CPU: {len(primes)} primes from {total_numbers} numbers in {cpu_time:.3f}s "
              f"({total_numbers/cpu_time:.0f} numbers/sec)")
        
        return primes
    
    def write_primes_to_csv(self, primes):
        if not primes:
            return
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            for prime in primes:
                writer.writerow([prime])
    
    def run_optimized_calculation(self):
        print(f"Начинаем вычисления с числа: {self.start_num}")
        print(f"Доступно OpenCL устройств: {len(self.contexts)}")
        for i, info in enumerate(self.device_info):
            print(f"  Устройство {i}: {info['vendor']} - {info['name']}")
        print(f"Доступно CPU ядер: {mp.cpu_count()}")
        print(f"Размер GPU батча: {GPU_BATCH_SIZE}")
        print(f"Размер CPU батча: {CPU_BATCH_SIZE}")
        
        n = self.start_num
        program_start_time = time.time()
        total_primes_found = 0
        
        while True:
            batch_start_time = time.time()
            all_primes = []
            
            # GPU вычисления в параллельных потоках
            gpu_threads = []
            gpu_results = [[] for _ in range(len(self.contexts))]
            
            for i in range(len(self.contexts)):
                range_end = n + GPU_BATCH_SIZE - 1
                thread = threading.Thread(
                    target=lambda idx=i, start=n, end=range_end: 
                    gpu_results[idx].extend(self.calculate_primes_gpu_batch(idx, start, end))
                )
                gpu_threads.append(thread)
                thread.start()
                n = range_end + 1
            
            # Ждем завершения GPU потоков
            for thread in gpu_threads:
                thread.join()
            
            # Собираем GPU результаты
            for result in gpu_results:
                all_primes.extend(result)
            
            # CPU вычисления
            cpu_range_end = n + CPU_BATCH_SIZE - 1
            cpu_primes = self.calculate_primes_cpu_parallel(n, cpu_range_end)
            all_primes.extend(cpu_primes)
            n = cpu_range_end + 1
            
            # Запись результатов
            if all_primes:
                all_primes.sort()
                self.write_primes_to_csv(all_primes)
                total_primes_found += len(all_primes)
            
            # Статистика
            batch_time = time.time() - batch_start_time
            total_time = time.time() - program_start_time
            
            print(f"БАТЧ: {batch_time:.2f}с | ВСЕГО: {total_time:.1f}с | "
                  f"ПРОСТЫХ в батче: {len(all_primes)} | Всего найдено: {total_primes_found} | ТЕКУЩЕЕ: {n}")
            print("-" * 80)

def main():
    print("PyOpenCL version:", cl.VERSION_TEXT)
    platforms = cl.get_platforms()
    print("Platforms:")
    for i, p in enumerate(platforms):
        print(f"  {i}: {p.name} ({p.vendor})")
        devices = p.get_devices()
        for j, d in enumerate(devices):
            print(f"    Device {j}: {d.name}")
    
    calculator = OptimizedPrimeCalculator()
    try:
        calculator.run_optimized_calculation()
    except KeyboardInterrupt:
        print("\nВычисление прервано пользователем")

if __name__ == "__main__":
    main()