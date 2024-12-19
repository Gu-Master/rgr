import numpy as np
import matplotlib.pyplot as plt

# Ввод данных
name_surname = "NIKITA GURACHEVSKII"
ascii_codes = [ord(char) for char in name_surname]
bit_sequence = []
for code in ascii_codes:
    bits = f"{code:08b}"
    bit_sequence.extend([int(bit) for bit in bits])

# Визуализация битовой последовательности
plt.figure(figsize=(10, 2))
plt.plot(bit_sequence, drawstyle='steps-mid')
plt.title("Битовая последовательность имени и фамилии")
plt.xlabel("Индекс")
plt.ylabel("Бит (0 или 1)")
plt.grid()
plt.show()

# Вычисление CRC
def calculate_crc(data_bits, poly_bits, crc_length):
    crc = [0] * crc_length
    data_bits_extended = data_bits + [0] * crc_length
    for i in range(len(data_bits)):
        if data_bits_extended[i] == 1:
            for j in range(len(poly_bits)):
                data_bits_extended[i + j] ^= poly_bits[j]
    return data_bits_extended[-crc_length:]

crc_length = 4
poly_bits = [1, 0, 1, 1]
crc = calculate_crc(bit_sequence, poly_bits, crc_length)
bit_sequence_with_crc = bit_sequence + crc

# Генерация последовательности Голда
def gold_sequence(length, taps1, taps2, init_state1, init_state2):
    def lfsr(taps, state):
        output = state[-1]
        new_bit = sum([state[tap - 1] for tap in taps]) % 2
        state = [new_bit] + state[:-1]
        return output, state

    seq1, seq2 = [], []
    state1, state2 = init_state1[:], init_state2[:]
    for _ in range(length):
        out1, state1 = lfsr(taps1, state1)
        out2, state2 = lfsr(taps2, state2)
        seq1.append(out1)
        seq2.append(out2)

    return [(seq1[i] ^ seq2[i]) for i in range(length)]

g_length = 31
gold_seq = gold_sequence(g_length, [5, 3], [5, 2], [1, 0, 0, 0, 1], [1, 1, 0, 0, 1])
full_sequence = gold_seq + bit_sequence_with_crc

# Временные отсчеты
n_samples_per_bit = 10
amplitude_signal = [bit for bit in full_sequence for _ in range(n_samples_per_bit)]

plt.figure(figsize=(10, 2))
plt.plot(amplitude_signal, drawstyle='steps-mid')
plt.title("Амплитудно-модулированный сигнал")
plt.xlabel("Индекс отсчета")
plt.ylabel("Амплитуда")
plt.grid()
plt.show()

# Создание нулевого массива и добавление сигнала
array_length = 2 * len(amplitude_signal)
zero_array = np.zeros(array_length)
position = int(input(f"Введите позицию (от 0 до {len(zero_array) - len(amplitude_signal)}): "))
zero_array[position:position + len(amplitude_signal)] = amplitude_signal

plt.figure(figsize=(10, 2))
plt.plot(zero_array, drawstyle='steps-mid')
plt.title("Сигнал в массиве")
plt.xlabel("Индекс")
plt.ylabel("Амплитуда")
plt.grid()
plt.show()

# Добавление шума
sigma = float(input("Введите значение σ для шума (например, 0.1): "))
noise = np.random.normal(0, sigma, size=len(zero_array))
noisy_signal = zero_array + noise

plt.figure(figsize=(10, 2))
plt.plot(noisy_signal)
plt.title("Зашумленный сигнал")
plt.xlabel("Индекс")
plt.ylabel("Амплитуда")
plt.grid()
plt.show()

# Корреляционный прием
def correlate_signal(signal, pattern):
    correlation = np.correlate(signal, pattern, mode="valid")
    start_index = np.argmax(correlation)
    return start_index

gold_samples = [bit for bit in gold_seq for _ in range(n_samples_per_bit)]
sync_start = correlate_signal(noisy_signal, gold_samples)
print(f"Синхронизация найдена на индексе: {sync_start}")

correlation = np.correlate(noisy_signal, gold_samples, mode="valid")
plt.figure(figsize=(10, 2))
plt.plot(correlation)
plt.title("Корреляция сигнала с последовательностью Голда")
plt.xlabel("Индекс")
plt.ylabel("Корреляция")
plt.grid()
plt.show()

# Демодуляция сигнала
def demodulate_signal(signal, n_samples, threshold):
    bits = []
    for i in range(0, len(signal), n_samples):
        segment = signal[i:i + n_samples]
        avg_amplitude = np.mean(segment)
        bits.append(1 if avg_amplitude > threshold else 0)
    return bits

threshold = 0.5
relevant_signal = noisy_signal[sync_start:]
demodulated_bits = demodulate_signal(relevant_signal, n_samples_per_bit, threshold)

plt.figure(figsize=(10, 2))
plt.plot(demodulated_bits, drawstyle='steps-mid')
plt.title("Демодулированные данные")
plt.xlabel("Индекс бита")
plt.ylabel("Бит (0 или 1)")
plt.grid()
plt.show()

# Проверка CRC
received_data = demodulated_bits[len(gold_seq):-crc_length]
received_crc = demodulated_bits[-crc_length:]
calculated_crc = calculate_crc(received_data, poly_bits, crc_length)

if received_crc == calculated_crc:
    print("CRC совпадает, ошибок нет.")
    decoded_bits = received_data
else:
    print("CRC не совпадает, данные повреждены.")
    decoded_bits = []
