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


x = [0, 0, 0, 0, 1]  # Порядковый номер 5 в двоичном формате
y = [1, 1, 0, 1, 0]  # x + 7 в двоичном формате
sequence_length = 31  # Длина последовательности Голда


gold_sequence = generate_gold_sequence(x, y, sequence_length)

# Функция корреляционного приема
def correlation_receiver(noisy_signal, sync_signal):
    # Вычисляем корреляцию между зашумленным сигналом и синхросигналом
    correlation = np.correlate(noisy_signal, sync_signal, mode='full')

    # Индекс максимальной корреляции - это точка синхронизации
    sync_start_index = np.argmax(correlation) - (len(sync_signal) - 1)

    # Обрезаем лишние биты до синхронизации
    synced_signal = noisy_signal[sync_start_index:]

    # Возвращаем индекс начала синхросигнала и обрезанный сигнал
    return sync_start_index, synced_signal





fs = 1000  # Частота дискретизации
Nx = 10  # Число отсчетов на один бит
L = len(gold_sequence)  # Длина последовательности данных
M = 7  # Длина CRC (если используется)
G = sequence_length  # Длина синхросигнала

#случайный сигнал
data_signal = np.random.randint(0, 2, size=L)  # случайный битовый сигнал данных


noisy_signal = np.concatenate([gold_sequence, data_signal])


sigma = float(input("Введите стандартное отклонение шума (σ): "))  # Ввод с клавиатуры
noise = np.random.normal(0, sigma, size=len(noisy_signal))


noisy_signal = noisy_signal + noise

# Применяем корреляционный прием к зашумленному сигналу
sync_start_index, synced_signal = correlation_receiver(noisy_signal, gold_sequence)

# Выводим индекс начала синхросигнала и длину обрезанного сигнала
print(f"Синхросигнал начинается с отсчета: {sync_start_index}")
print(f"Длина обрезанного сигнала: {len(synced_signal)}")


plt.figure(figsize=(12, 4))
plt.plot(noisy_signal, label="Зашумленный сигнал", color='r')
plt.axvline(x=sync_start_index, color='b', linestyle='--', label="Начало синхросигнала")
plt.title("Зашумленный сигнал с определением начала синхросигнала")
plt.xlabel("Время (отсчеты)")
plt.ylabel("Амплитуда")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(synced_signal, label="Синхронизированный сигнал", color='g')
plt.title("Обрезанный сигнал после синхронизации")
plt.xlabel("Время (отсчеты)")
plt.ylabel("Амплитуда")
plt.grid()
plt.legend()
plt.show()
