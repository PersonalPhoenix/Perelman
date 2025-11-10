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
GPU_BATCH_SIZE = 500000    # 500K чисел для GPU
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

# Оптимизированное ядро OpenCL
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
    
    uint sqrt_n = sqrt((float)n);
    for (uint i = 3; i <= sqrt_n; i += 2) {
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
        
        for platform in platforms:
            try:
                # Пробуем использовать все GPU устройства
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                for device in devices:
                    try:
                        ctx = cl.Context([device])
                        queue = cl.CommandQueue(ctx)
                        program = cl.Program(ctx, opencl_kernel_code).build()
                        
                        self.contexts.append(ctx)
                        self.queues.append(queue)
                        self.programs.append(program)
                        print(f"Успешно инициализировано: {device.vendor} - {device.name}")
                        print(f"  Макс. рабочая группа: {device.max_work_group_size}")
                        print(f"  Выч. единицы: {device.max_compute_units}")
                    except Exception as e:
                        print(f"Ошибка инициализации {device.name}: {e}")
            except:
                continue
        
        if not self.contexts:
            print("Не найдено GPU устройств, используем только CPU")
    
    def calculate_primes_gpu_batch(self, device_idx, start, end):
        """GPU вычисления в основном потоке"""
        if device_idx >= len(self.contexts):
            return []
        
        ctx = self.contexts[device_idx]
        queue = self.queues[device_idx]
        program = self.programs[device_idx]
        
        numbers = np.arange(start, end + 1, dtype=np.uint32)
        total_numbers = len(numbers)
        
        # Оптимизация: используем меньший размер рабочей группы
        device = ctx.devices[0]
        max_wg_size = device.max_work_group_size
        work_group_size = min(256, max_wg_size)
        
        numbers_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=numbers)
        results_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, numbers.nbytes)
        
        # Запускаем с оптимальным размером рабочей группы
        global_size = ((total_numbers + work_group_size - 1) // work_group_size) * work_group_size
        
        start_time = time.time()
        program.find_primes(queue, (global_size,), (work_group_size,), 
                          numbers_buf, results_buf, np.uint32(total_numbers))
        
        results = np.empty(total_numbers, dtype=np.uint32)
        cl.enqueue_copy(queue, results, results_buf).wait()
        gpu_time = time.time() - start_time
        
        primes = numbers[results == 1]
        
        print(f"GPU {device_idx}: {len(primes)} primes from {total_numbers} numbers in {gpu_time:.3f}s "
              f"({total_numbers/gpu_time:.0f} numbers/sec)")
        
        return primes.tolist()
    
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
        print(f"Доступно CPU ядер: {mp.cpu_count()}")
        print(f"Размер GPU батча: {GPU_BATCH_SIZE}")
        print(f"Размер CPU батча: {CPU_BATCH_SIZE}")
        
        n = self.start_num
        program_start_time = time.time()
        
        while True:
            batch_start_time = time.time()
            all_primes = []
            
            # GPU вычисления в основном потоке
            gpu_results = []
            if self.contexts:
                # Создаем потоки для параллельного выполнения GPU вычислений
                threads = []
                for i in range(len(self.contexts)):
                    range_end = n + GPU_BATCH_SIZE - 1
                    thread = threading.Thread(
                        target=lambda idx=i, start=n, end=range_end: 
                        gpu_results.append((idx, self.calculate_primes_gpu_batch(idx, start, end)))
                    )
                    threads.append(thread)
                    n = range_end + 1
                
                # Запускаем все потоки
                for thread in threads:
                    thread.start()
                
                # Ждем завершения всех потоков
                for thread in threads:
                    thread.join()
                
                # Собираем результаты
                for device_idx, primes in gpu_results:
                    if primes:
                        all_primes.extend(primes)
            
            # CPU вычисления
            cpu_range_end = n + CPU_BATCH_SIZE - 1
            cpu_primes = self.calculate_primes_cpu_parallel(n, cpu_range_end)
            all_primes.extend(cpu_primes)
            n = cpu_range_end + 1
            
            # Запись результатов
            if all_primes:
                all_primes.sort()
                self.write_primes_to_csv(all_primes)
            
            # Статистика
            batch_time = time.time() - batch_start_time
            total_time = time.time() - program_start_time
            
            print(f"БАТЧ: {batch_time:.2f}с | ВСЕГО: {total_time:.1f}с | "
                  f"ПРОСТЫХ: {len(all_primes)} | ТЕКУЩЕЕ: {n}")
            print("-" * 80)

def main():
    print("PyOpenCL version:", cl.VERSION_TEXT)
    print("Platforms:", [f"{p.name} ({p.vendor})" for p in cl.get_platforms()])
    
    calculator = OptimizedPrimeCalculator()
    try:
        calculator.run_optimized_calculation()
    except KeyboardInterrupt:
        print("\nВычисление прервано пользователем")

if __name__ == "__main__":
    main()