import numpy as np
import matplotlib.pyplot as plt

# Преобразование строки в битовую последовательность
name_surname = "NIKITA GURACHEVSKII"
ascii_codes = [ord(char) for char in name_surname]
bit_sequence = []
for code in ascii_codes:
    bits = f"{code:08b}"
    bit_sequence.extend([int(bit) for bit in bits])

# Функция для расчета CRC
def calculate_crc(data_bits, poly_bits, crc_length):
    crc = [0] * crc_length
    data_bits_extended = data_bits + [0] * crc_length
    for i in range(len(data_bits)):
        if data_bits_extended[i] == 1:
            for j in range(len(poly_bits)):
                data_bits_extended[i + j] ^= poly_bits[j]
    return data_bits_extended[-crc_length:]

# Задание параметров CRC
crc_length = 7
poly_bits = [1, 1, 1, 1, 1, 0, 1, 1]  # Полином x^7 + x^6 + x^5 + x^4 + x^3 + x + 1

# Расчет CRC
crc = calculate_crc(bit_sequence, poly_bits, crc_length)

# Генерация последовательности Голда
def generate_gold_sequence(x, y, length):
    sequence = []
    for _ in range(length):
        bit_x = (x[4] + x[2]) % 2
        bit_y = (y[4] + y[3] + y[2] + y[0]) % 2
        sequence.append((x[4] + y[4]) % 2)
        shift_register(x, bit_x)
        shift_register(y, bit_y)
    return sequence

def shift_register(register, new_bit):
    for i in range(len(register) - 1):
        register[i] = register[i + 1]
    register[-1] = new_bit

x = [0, 0, 0, 0, 1]
y = [1, 1, 0, 1, 0]
gold_sequence = generate_gold_sequence(x, y, 31)

# Объединение всех последовательностей: данные, CRC, Голда
combined_sequence = bit_sequence + crc + gold_sequence

# Преобразование битов в временные отсчеты
def bits_to_signal(bits, N):
    signal = []
    for bit in bits:
        signal.extend([bit] * N)
    return signal

# Преобразование в временные отсчеты
N = 5  # Количество отсчетов на бит
signal = bits_to_signal(combined_sequence, N)

# Создание нулевого массива длиной 2*N*(L+M+G)
L = len(bit_sequence)    # Длина данных
M = len(crc)             # Длина CRC
G = len(gold_sequence)   # Длина Голда

total_length = 2 * N * (L + M + G)
zero_array = [0] * total_length

# Ввод от пользователя
index = int(input(f"Введите число от 0 до {total_length - 1}: "))

# Проверка корректности ввода
if 0 <= index < total_length:
    # Вставляем временные отсчеты в массив
    zero_array[index:index + len(signal)] = signal
else:
    print("Введенное число не в пределах допустимого диапазона!")

# Ввод параметра σ для шума
sigma = float(input("Введите значение стандартного отклонения шума (σ): "))

# Генерация шума с нормальным распределением (μ=0, σ=std)
noise = np.random.normal(0, sigma, total_length)

# Добавление шума к сигналу
noisy_signal = np.array(zero_array) + noise

# Визуализация зашумленного принятого сигнала
plt.figure(figsize=(12, 4))
plt.plot(noisy_signal, label="Зашумленный сигнал", color='r')
plt.title("Зашумленный принятый сигнал")
plt.xlabel("Время (отсчеты)")
plt.ylabel("Амплитуда")
plt.grid()
plt.legend()
plt.show()

# Результат
print(f"Длина зашумленного сигнала: {len(noisy_signal)}")
