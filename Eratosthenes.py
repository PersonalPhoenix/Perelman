import math
import csv
import os
import time

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

filename = 'primes.csv'
start_num = 2
primes = []

if os.path.exists(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                try:
                    primes.append(int(row[0]))
                except ValueError:
                    pass
    if primes:
        start_num = primes[-1] + 1

program_start_time = time.time()
last_prime_time = program_start_time

with open(filename, 'a', newline='') as f:
    writer = csv.writer(f)
    n = start_num
    while True:
        start_time = time.time()
        if is_prime(n):
            primes.append(n)
            writer.writerow([n])
            current_time = time.time()
            prime_time = current_time - start_time
            total_time = current_time - program_start_time
            elapsed_since_last = current_time - last_prime_time
            avg_time_per_prime = total_time / len(primes)
            eta = avg_time_per_prime * (len(primes) + 1) - total_time
            
            print(f"Найдено простое: {n} | "
                  f"Время проверки: {prime_time:.3f}с | "
                  f"Общее время: {total_time:.1f}с | "
                  f"ETA: {eta:.1f}с")
            
            last_prime_time = current_time
        n += 1