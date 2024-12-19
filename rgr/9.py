import numpy as np
import matplotlib.pyplot as plt


# Функция для принятия решения по каждому символу
def decision_function(signal, N, P):
    """
    Функция принимает решение по каждому N отсчетам, где N - длина одного символа
    и P - пороговое значение для интерпретации 0 или 1.

    signal: сигнал, в котором необходимо разобрать символы.
    N: количество отсчетов на один символ.
    P: пороговое значение для определения 0 или 1.

    Возвращает массив битов, где 0 - это символ, который интерпретируется как 0,
    а 1 - как 1.
    """
    bits = []
    for i in range(0, len(signal), N):
        symbol = signal[i:i + N]
        avg_value = np.mean(symbol)  # Среднее значение символа
        if avg_value > P:
            bits.append(1)  # если среднее значение больше порога, то это 1
        else:
            bits.append(0)  # если меньше, то это 0
    return bits




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



# Для примера создадим случайный сигнал с шумом
fs = 1000  # Частота дискретизации
Nx = 10  # Число отсчетов на один бит
L = 10  # Длина последовательности данных (просто для примера)
M = 7  # Длина CRC (если используется)
G = 31  # Длина синхросигнала

# Создаем сигнал: сначала последовательность Голда, потом данные и CRC
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
P = 0.5  # Порог, можно настроить в зависимости от данных

# Применяем функцию для принятия решения
decoded_bits = decision_function(noisy_signal, Nx, P)

# Выводим результаты
print("Декодированные биты:", decoded_bits)

# Визуализируем сигнал и принятые решения
plt.figure(figsize=(12, 4))
plt.plot(noisy_signal, label="Зашумленный сигнал")
plt.title("Зашумленный сигнал и принятие решений")
plt.xlabel("Время (отсчеты)")
plt.ylabel("Амплитуда")
plt.grid()

# Помечаем результаты решений
for i in range(len(decoded_bits)):
    plt.axvline(x=i * Nx, color='r', linestyle='--', label=f"Решение: {decoded_bits[i]}")
plt.legend()
plt.show()
