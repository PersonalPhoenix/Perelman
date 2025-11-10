import pyopencl as cl
import numpy as np
import csv
import os
import time
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor



# Конфигурация
CSV_FILENAME = 'primes.csv'
BATCH_SIZE = 10000  # Размер батча для GPU
CPU_BATCH_SIZE = 5000  # Размер батча для CPU

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

# Функция для выбора устройств вычисления
def setup_opencl_devices():
    platforms = cl.get_platforms()
    devices = []
    for p in platforms:
        try:
            devs = p.get_devices(device_type=cl.device_type.GPU)
            for d in devs:
                if 'AMD' in d.vendor or 'Intel' in d.vendor:
                    devices.append((p, d))
                    print(f"Найдено устройство: {d.vendor} - {d.name}")
        except:
            continue
    return devices

# OpenCL kernel код
opencl_kernel_code = """
__kernel void find_primes(__global const uint* numbers, 
                          __global uint* results,
                          const uint count) {
    uint gid = get_global_id(0);
    if (gid >= count) return;
    
    uint n = numbers[gid];
    uint result = 1;
    
    if (n < 2) {
        result = 0;
    } else if (n == 2) {
        result = 1;
    } else if (n % 2 == 0) {
        result = 0;
    } else {
        for (uint i = 3; i * i <= n; i += 2) {
            if (n % i == 0) {
                result = 0;
                break;
            }
        }
    }
    
    results[gid] = result;
}
"""

class PrimeCalculator:
    def __init__(self):
        self.filename = CSV_FILENAME
        self.start_num = get_last_prime_from_csv(self.filename) + 1
        self.opencl_devices = setup_opencl_devices()
        self.opencl_contexts = []
        self.opencl_queues = []
        self.opencl_programs = []
        
        # Инициализация OpenCL
        for platform, device in self.opencl_devices:
            try:
                ctx = cl.Context([device])
                queue = cl.CommandQueue(ctx)
                program = cl.Program(ctx, opencl_kernel_code).build()
                
                self.opencl_contexts.append(ctx)
                self.opencl_queues.append(queue)
                self.opencl_programs.append(program)
                print(f"Успешно инициализировано устройство: {device.name}")
            except Exception as e:
                print(f"Ошибка инициализации устройства {device.name}: {e}")
    
    def calculate_primes_gpu(self, device_idx, start, end):
        if device_idx >= len(self.opencl_contexts):
            return []
        
        ctx = self.opencl_contexts[device_idx]
        queue = self.opencl_queues[device_idx]
        program = self.opencl_programs[device_idx]
        
        numbers = np.arange(start, end + 1, dtype=np.uint32)
        total_numbers = len(numbers)
        
        numbers_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=numbers)
        results_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, numbers.nbytes)
        
        program.find_primes(queue, (total_numbers,), None, numbers_buf, results_buf, np.uint32(total_numbers))
        
        results = np.empty(total_numbers, dtype=np.uint32)
        cl.enqueue_copy(queue, results, results_buf).wait()
        
        primes = numbers[results == 1]
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
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(cpu_prime_worker, ranges))
        
        primes = []
        for result in results:
            primes.extend(result)
        
        return primes
    
    def write_primes_to_csv(self, primes):
        if not primes:
            return
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            for prime in primes:
                writer.writerow([prime])
    
    def run_distributed_calculation(self):
        print(f"Начинаем вычисления с числа: {self.start_num}")
        print(f"Доступно OpenCL устройств: {len(self.opencl_contexts)}")
        print(f"Доступно CPU ядер: {mp.cpu_count()}")
        
        n = self.start_num
        program_start_time = time.time()
        
        while True:
            batch_start_time = time.time()
            all_primes = []
            
            # GPU вычисления
            gpu_ranges = []
            for i in range(len(self.opencl_contexts)):
                range_end = n + BATCH_SIZE - 1
                gpu_ranges.append((i, n, range_end))
                n = range_end + 1
            
            # Запуск GPU вычислений
            for device_idx, start, end in gpu_ranges:
                try:
                    primes = self.calculate_primes_gpu(device_idx, start, end)
                    all_primes.extend(primes)
                    print(f"GPU {device_idx}: обработан диапазон {start}-{end}, найдено {len(primes)} простых чисел")
                except Exception as e:
                    print(f"Ошибка на GPU {device_idx}: {e}")
            
            # CPU вычисления
            cpu_range_end = n + CPU_BATCH_SIZE - 1
            cpu_primes = self.calculate_primes_cpu_parallel(n, cpu_range_end)
            all_primes.extend(cpu_primes)
            print(f"CPU: обработан диапазон {n}-{cpu_range_end}, найдено {len(cpu_primes)} простых чисел")
            
            n = cpu_range_end + 1
            
            # Запись результатов
            if all_primes:
                all_primes.sort()
                self.write_primes_to_csv(all_primes)
            
            # Статистика
            batch_time = time.time() - batch_start_time
            total_time = time.time() - program_start_time
            total_primes = len(all_primes) if all_primes else 0
            
            print(f"Батч завершен за {batch_time:.2f}с | "
                  f"Всего времени: {total_time:.1f}с | "
                  f"Найдено простых: {total_primes}")
            
            # Небольшая пауза для предотвращения перегрева
            time.sleep(0.1)

def main():
    print("PyOpenCL version:", cl.VERSION_TEXT)
    print("Platforms:", cl.get_platforms())
    calculator = PrimeCalculator()
    try:
        calculator.run_distributed_calculation()
    except KeyboardInterrupt:
        print("\nВычисление прервано пользователем")

if __name__ == "__main__":
    main()