import matplotlib.pyplot as plt
import numpy as np
import random


# Функция преобразования строки в битовую последовательность
def string_to_bits(s):
    result = []
    for char in s:
        bits = bin(ord(char))[2:].zfill(8)  # Преобразуем каждый символ в 8-битное двоичное представление
        result.extend([int(bit) for bit in bits])
    return result


# Функция для вычисления CRC с использованием заданного генератора
def calculate_crc(data, generator):
    data_size = len(data)
    generator_size = len(generator)
    extended_data = data + [0] * (generator_size - 1)  # Дополняем данные нулями для вычисления CRC

    for i in range(data_size):
        if extended_data[i] == 1:  # Если старший бит равен 1, применяем генератор
            for j in range(generator_size):
                extended_data[i + j] ^= generator[j]

    crc = extended_data[data_size:]  # CRC будет в конце дополненных данных
    return crc


# Функция преобразования битов в строку
def bits_to_string(bits):
    string = ""
    for i in range(0, len(bits), 8):
        byte = bits[i:i + 8]  # Группируем биты по 8
        char_code = int("".join(str(bit) for bit in byte), 2)  # Преобразуем в символ
        string += chr(char_code)
    return string


# Функция визуализации спектра сигнала
def visualize_spectrum(signal, fs, title="Signal Spectrum"):
    frequencies = np.fft.fftfreq(len(signal), 1 / fs)  # Частоты для спектра
    spectrum = np.abs(np.fft.fft(signal))  # Преобразование Фурье сигнала

    plt.figure(figsize=(10, 4))
    plt.plot(frequencies, spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.grid(True)
    plt.show()


# Функция визуализации битовой последовательности
def visualize_bits(bits, title="Bit Visualization"):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.step(range(len(bits)), bits, where='post')

    min_val = min(bits)
    max_val = max(bits)
    if min_val != max_val:
        ax.set_ylim([min_val - 0.5, max_val + 0.5])

    ax.set_xlabel("Bit Index")
    ax.set_ylabel("Bit Value")
    ax.set_title(title)
    ax.grid(True)
    plt.show()


# 12) Удаление CRC и восстановление текста
if __name__ == "__main__":
    # Вводим данные
    name = input("Введите ваше имя (латинскими буквами): ")
    surname = input("Введите вашу фамилию (латинскими буквами): ")

    full_name = name + surname
    bit_sequence = string_to_bits(full_name)

    print("\nИсходная битовая последовательность:", bit_sequence)

    # CRC генератор
    generator = [1, 0, 1, 1, 1, 0, 1, 1]
    crc = calculate_crc(bit_sequence, generator)
    bit_sequence_with_crc = bit_sequence + crc
    print("\nБитовая последовательность с CRC:", bit_sequence_with_crc)

    # 12) Удаляем CRC и восстанавливаем текст
    decoded_bits_without_crc = bit_sequence_with_crc[:-len(crc)]  # Убираем CRC

    decoded_string = bits_to_string(decoded_bits_without_crc)  # Восстанавливаем строку
    print("\nДекодированный текст:", decoded_string)

    # 13) Визуализация спектра для различных длительностей символа
    fs = 100  # частота дискретизации
    samples_per_bit_values = [5, 10, 20]  # длительности символов

    for samples_per_bit in samples_per_bit_values:
        # Генерация временных выборок
        time_samples = []
        for bit in bit_sequence_with_crc:
            time_samples.extend([bit] * samples_per_bit)

        # Визуализация спектра
        visualize_spectrum(time_samples, fs, f"Спектр сигнала (N={samples_per_bit})")

        # Зашумленный сигнал
        noise = np.random.normal(0, 0.5, len(time_samples))  # Добавление шума
        noisy_signal = (np.array(time_samples) + noise).tolist()
        visualize_spectrum(noisy_signal, fs, f"Спектр зашумленного сигнала (N={samples_per_bit})")

    # Визуализация всех спектров на одном графике
    plt.figure(figsize=(12, 6))

    for samples_per_bit in samples_per_bit_values:
        time_samples = []
        for bit in bit_sequence_with_crc:
            time_samples.extend([bit] * samples_per_bit)

        # Оригинальный сигнал
        plt.subplot(3, 1, 1)
        visualize_spectrum(time_samples, fs, f"Спектр сигнала (N={samples_per_bit})")

        # Зашумленный сигнал
        noise = np.random.normal(0, 0.5, len(time_samples))
        noisy_signal = (np.array(time_samples) + noise).tolist()
        plt.subplot(3, 1, 2)
        visualize_spectrum(noisy_signal, fs, f"Спектр зашумленного сигнала (N={samples_per_bit})")

    plt.tight_layout()
    plt.show()
