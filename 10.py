import numpy as np
import matplotlib.pyplot as plt

def shift_register(register, new_bit):
    for i in range(len(register) - 1):
        register[i] = register[i + 1]
    register[-1] = new_bit

def generate_gold_sequence(x, y, length):
    sequence = []
    for _ in range(length):
        bit_x = (x[4] + x[2]) % 2
        bit_y = (y[4] + y[3] + y[2] + y[0]) % 2
        sequence.append((x[4] + y[4]) % 2)
        shift_register(x, bit_x)
        shift_register(y, bit_y)
    return sequence

def decision_function(signal, N, P):
    """
    Принимает решение (0 или 1) по каждому N отсчету на основе порога P.

    signal: зашумленный сигнал.
    N: число отсчетов на один символ.
    P: пороговое значение для принятия решения.

    Возвращает список битов, представляющих данные.
    """
    decoded_bits = []
    for i in range(0, len(signal), N):
        sample = signal[i:i + N]
        if len(sample) < N:
            continue  # Пропускаем последние неполные символы
        avg = np.mean(sample)  # Среднее значение для блока отсчетов
        print(f"Блок {i//N}: Среднее значение = {avg}")
        if avg > P:
            decoded_bits.append(1)  # Если среднее больше порога, то 1
        else:
            decoded_bits.append(0)  # Иначе 0
    return decoded_bits

def remove_sync_sequence(signal, G):
    """
    Удаляет G бит последовательности синхронизации из начала сигнала.
    """
    return signal[G:]

# Пример использования:

fs = 1000  # Частота дискретизации
Nx = 10  # Число отсчетов на один бит
L = 10  # Длина последовательности данных
M = 7  # Длина CRC (если используется)
G = 31  # Длина синхросигнала

x = [0, 0, 0, 0, 1]
y = [1, 1, 0, 1, 0]
gold_sequence = generate_gold_sequence(x, y, G)

data_signal = np.random.randint(0, 2, size=L)  # случайный битовый сигнал данных
signal = np.concatenate([gold_sequence, data_signal])  # объединяем сигнал

# Добавление шума с нормальным распределением
sigma = 0.1  # стандартное отклонение шума
noise = np.random.normal(0, sigma, size=len(signal))
noisy_signal = signal + noise

# Пороговое значение для принятия решения
P = 0.5  # Порог

# Применяем функцию для принятия решения
decoded_bits = decision_function(noisy_signal, Nx, P)

# Удаляем синхросигнал
cleaned_signal = remove_sync_sequence(decoded_bits, G)

# Выводим результаты
print("Декодированные биты без синхросигнала:", cleaned_signal)

# Визуализируем сигнал до и после удаления синхросигнала
plt.figure(figsize=(12, 4))
plt.plot(noisy_signal, label="Зашумленный сигнал")
plt.axvline(x=G * Nx, color='r', linestyle='--', label=f"Удален синхросигнал ({G} бит)")

plt.title("Зашумленный сигнал и удаление синхросигнала")
plt.xlabel("Время (отсчеты)")
plt.ylabel("Амплитуда")
plt.grid()
plt.legend()
plt.show()
