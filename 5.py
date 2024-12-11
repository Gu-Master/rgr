import time
import matplotlib.pyplot as plt
import numpy as np
import crc32c



def generate_bit_sequence(length):
    return bytes(np.random.randint(0, 256, length // 8))



def calculate_crc32(data):
    # Pre-compute the CRC lookup table
    crc_table = [0] * 256
    for i in range(256):
        crc = i
        for j in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc = crc >> 1
        crc_table[i] = crc

    crc = 0xFFFFFFFF
    for byte in data:
        crc = crc_table[(crc ^ byte) & 0xFF] ^ (crc >> 8)

    # Invert the final CRC value
    crc ^= 0xFFFFFFFF

    return crc



def calculate_crc8(data):
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc <<= 1
            crc &= 0xFF  # ограничиваем до 8 бит
    return crc



def calculate_crc16(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x8005
            else:
                crc <<= 1
            crc &= 0xFFFF  # ограничиваем до 16 бит
    return crc



bit_lengths = [2**i for i in range(7, 20)]  # от 128 до 524288


polynomials = {
    "CRC-8": calculate_crc8,
    "CRC-16": calculate_crc16,
    "CRC-32": calculate_crc32
}


execution_times_crc32c = {poly: [] for poly in polynomials}


execution_times_manual = {poly: [] for poly in polynomials}


for poly_name, calculate_crc in polynomials.items():
    for length in bit_lengths:
        data = generate_bit_sequence(length)


        start_time = time.perf_counter()
        for _ in range(10):  # Усредняем по 10 итераций
            crc = crc32c.crc32c(data)
        end_time = time.perf_counter()
        execution_times_crc32c[poly_name].append((end_time - start_time) / 10)


        start_time = time.perf_counter()
        for _ in range(10):  # Усредняем по 10 итераций
            crc = calculate_crc(data)
        end_time = time.perf_counter()
        execution_times_manual[poly_name].append((end_time - start_time) / 10)


plt.figure(figsize=(14, 7))


plt.subplot(1, 2, 1)
for poly in polynomials:
    plt.plot(bit_lengths, execution_times_crc32c[poly], label=f"Полином {poly} (crc32c)")
plt.title("CRC с использованием crc32c")
plt.xlabel("Длина битовой последовательности")
plt.ylabel("Время выполнения (секунды)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)


plt.subplot(1, 2, 2)
for poly in polynomials:
    plt.plot(bit_lengths, execution_times_manual[poly], label=f"Полином {poly} (ручная реализация)")
plt.title("CRC с ручной реализацией")
plt.xlabel("Длина битовой последовательности")
plt.ylabel("Время выполнения (секунды)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.show()
