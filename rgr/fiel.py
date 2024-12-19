import numpy as np
import matplotlib.pyplot as plt

name_surname = "NIKITA GURACHEVSKII"

# Преобразование строки в ASCII-код
ascii_codes = [ord(char) for char in name_surname]

# Преобразование ASCII-кода в битовую последовательность
bit_sequence = []
for code in ascii_codes:
    bits = f"{code:08b}"  # Преобразование в 8-битное представление
    bit_sequence.extend([int(bit) for bit in bits])

print("Сформированная битовая последовательность:")
print(bit_sequence)

# Функция для расчета CRC
def calculate_crc(data_bits, poly_bits, crc_length):
    crc = [0] * crc_length
    data_bits_extended = data_bits + [0] * crc_length
    for i in range(len(data_bits)):
        if data_bits_extended[i] == 1:  # XOR с полиномом, если старший бит равен 1
            for j in range(len(poly_bits)):
                data_bits_extended[i + j] ^= poly_bits[j]
    return data_bits_extended[-crc_length:]

# Задание параметров CRC
crc_length = 7  # Длина CRC соответствует старшему порядку полинома
poly_bits = [1, 1, 1, 1, 1, 0, 1, 1]  # Полином x^7 + x^6 + x^5 + x^4 + x^3 + x + 1

# Расчет CRC
crc = calculate_crc(bit_sequence, poly_bits, crc_length)

# Объединение битовой последовательности с CRC
bit_sequence_with_crc = bit_sequence + crc

# Вывод результата
print("\nРазработанный ASCII-кодер:")
print(f"Имя и фамилия: {name_surname}")
print(f"ASCII-коды символов: {ascii_codes}")
print(f"Битовая последовательность (L={len(bit_sequence)}): {bit_sequence}")
print(f"CRC (M={crc_length}): {crc}")
print(f"Битовая последовательность с CRC (L+M={len(bit_sequence_with_crc)}): {bit_sequence_with_crc}")



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

# Сдвиг регистра
def shift_register(register, new_bit):
    for i in range(len(register) - 1):
        register[i] = register[i + 1]
    register[-1] = new_bit

# Функция для автокорреляции
def autocorrelation(sequence):
    length = len(sequence)
    result = []
    for shift in range(length):
        shifted_sequence = np.roll(sequence, shift)
        matches = sum(1 for i in range(length) if sequence[i] == shifted_sequence[i])
        result.append(2.0 * matches / length - 1.0)
    return result

# Исходные данные
x = [0, 0, 0, 0, 1]  # Порядковый номер 5 в двоичном формате
y = [1, 1, 0, 1, 0]  # x + 7 в двоичном формате
sequence_length = 31  # Длина последовательности Голда

# Генерация последовательности Голда
gold_sequence = generate_gold_sequence(x, y, sequence_length)

# Визуализация последовательности Голда
plt.figure(figsize=(12, 4))
plt.stem(gold_sequence)
plt.title("Последовательность Голда")
plt.xlabel("Индекс")
plt.ylabel("Значение")
plt.grid()
plt.show()


combined_bits = gold_sequence + bit_sequence  + crc

# Визуализация данных с последовательностью Голда
plt.figure(figsize=(12, 4))
plt.stem(combined_bits)
plt.title("Битовая последовательность с синхронизацией (Голда)")
plt.xlabel("Индекс")
plt.ylabel("Значение")
plt.grid()
plt.show()

# Расчет и вывод автокорреляции
autocorr = autocorrelation(gold_sequence)
plt.figure(figsize=(12, 4))
plt.stem(autocorr)
plt.title("Автокорреляция последовательности Голда")
plt.xlabel("Сдвиг")
plt.ylabel("Значение автокорреляции")
plt.grid()
plt.show()

print("Генерация и добавление последовательности Голда завершено.")
