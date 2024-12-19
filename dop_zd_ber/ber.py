import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack


def ascii_coder(text):
    bit_sequence = []
    for symbol in text:
        ascii_symbol = ord(symbol)
        bits_symbol = bin(ascii_symbol)[2:].zfill(8)
        bit_sequence.extend(int(bit) for bit in bits_symbol)
    return bit_sequence


def computeCRC(packet, polynomial):
    packet_zeros = packet[:] + [0] * (len(polynomial) - 1)
    for i in range(len(packet)):
        if packet_zeros[i] == 1:
            for j in range(len(polynomial)):
                packet_zeros[i + j] ^= polynomial[j]

    CRC = packet_zeros[-(len(polynomial) - 1):]
    return CRC


def create_gold_sequence(x, y, len_sequence):
    gold_sequence = []

    for i in range(len_sequence):
        xor_shift_x = x[2] ^ x[4]
        xor_shift_y = y[2] ^ y[4]

        gold_sequence.append(x[-1] ^ y[-1])

        x.pop()
        y.pop()

        x.insert(0, xor_shift_x)
        y.insert(0, xor_shift_y)

    return gold_sequence


def bits_to_samples(bit_sequence, N):
    signal_samples = []
    for bit in bit_sequence:
        signal_samples.extend([bit] * N)
    return signal_samples


# Receiver
def NormalizedCorrelation(x, y):
    corr_array = []
    for i in range(len(x) - len(y) + 1):
        sumXY = 0.0
        sumX2 = 0.0
        sumY2 = 0.0
        corr = 0.0
        shifted_sequence = x[i:i + len(y)]
        for j in range(len(shifted_sequence)):
            sumXY += shifted_sequence[j] * y[j]
            sumX2 += shifted_sequence[j] * shifted_sequence[j]
            sumY2 += y[j] * y[j]
        corr = sumXY / np.sqrt(sumX2 * sumY2)
        corr_array.append(corr)
    return corr_array


def samples_to_bits(signal_samples, N):
    bit_sequence = []
    P = 0.5
    num_blocks = len(signal_samples) // N
    for i in range(num_blocks):
        block = signal_samples[i * N:(i + 1) * N]
        mean = np.mean(block)
        if mean >= P:
            bit_sequence.append(1)
        else:
            bit_sequence.append(0)

    return bit_sequence


def samples_to_bits_4(signal_samples, N):
    bit_sequence = []
    num_blocks = len(signal_samples) // N

    for i in range(num_blocks):
        block = signal_samples[i * N:(i + 1) * N]
        mean = np.mean(block)

        if 0 <= mean < 0.25:
            bit_sequence.append(0)
        elif 0.25 <= mean < 0.5:
            bit_sequence.append(0.33)
        elif 0.5 <= mean < 0.75:
            bit_sequence.append(0.66)
        else:
            bit_sequence.append(1)

    return bit_sequence


def check_packet(received_packet, polynomial):
    result = computeCRC(received_packet, polynomial)

    return all(bit == 0 for bit in result)


def ascii_decoder(bit_sequence):
    text = ''
    for i in range(0, len(bit_sequence), 8):
        bits_symbol = bit_sequence[i:i + 8]
        if len(bits_symbol) < 8:
            break
        ascii_symbol = int(''.join(map(str, bits_symbol)), 2)
        text += chr(ascii_symbol)
    return text


# Spectrum
def plot_spectrum(signal, name):
    spectrum = fftpack.fft(signal)
    freqs = np.arange(0, 1, 1 / len(signal))

    plt.plot(freqs, np.abs(spectrum), label=name)


def compute_ber(original_bits, received_bits):
    errors = sum([1 for ob, rb in zip(original_bits, received_bits) if ob != rb])
    return errors / len(original_bits)


def amplitude4(bit_1, bit_2):
    if bit_1 == 0:
        if bit_2 == 0:
            return -1.5

        elif bit_2 == 1:
            return -0.5

    elif bit_1 == 1:
        if bit_2 == 0:
            return 0.5

        elif bit_2 == 1:
            return 1.5


def decode_noisy_signal_two(noisy_signal, P=0.5):
    bit_sequence = []

    for sample in noisy_signal:
        if sample >= P:
            bit_sequence.append(1)
        else:
            bit_sequence.append(0)

    return bit_sequence


def decode_noisy_signal_four(noisy_signal):
    bit_sequence = []

    for sample in noisy_signal:
        if sample < -1:
            bit_sequence.append(-1.5)
        elif -1 <= sample < 0:
            bit_sequence.append(-0.5)
        elif 0 <= sample < 1:
            bit_sequence.append(0.5)
        else:
            bit_sequence.append(1.5)

    return bit_sequence



def correlation_receiver(x, y, stop_word):
    corr_array = NormalizedCorrelation(x, y)
    start_useful_bits = np.argmax(corr_array)
    print(f"Индекс начала полезного сигнала: {start_useful_bits}")
    corrected_signal = x[start_useful_bits:]

    corrected_signal = corrected_signal[:stop_word]

    return corrected_signal


# 1)
name = input("Введите имя на латинице: ")
surname = input("Введите фамилию на латинице: ")
# 2)
name_surname = name + " " + surname
bit_sequence = ascii_coder(name_surname)

# 3)
G = [1, 1, 0, 1, 1, 1, 1, 0]
CRC = computeCRC(bit_sequence, G)
CRC_print = ''.join(map(str, CRC))
print(f"CRC для битовой последовательности с данными: {CRC_print}")

bit_sequence_crc = bit_sequence + CRC

# 4)
x = [0, 1, 1, 0, 0]
y = [1, 0, 0, 1, 1]
len_sequence = 31

gold_sequence = create_gold_sequence(x, y, len_sequence)
bit_sequence_crc_gold = gold_sequence + bit_sequence_crc



stop_word = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0]
bit_sequence_crc_gold_stop = bit_sequence_crc_gold + stop_word

# 5)
N = 10
signal_samples = bits_to_samples(bit_sequence_crc_gold_stop, N)

# 6)
signal = [0] * (2 * len(signal_samples))
position = int(input(
    f"Введите номер позиции для вставки битовой последовательности (от 0 до {len(signal_samples)}): "))
insert_length = min(len(signal_samples), (2 * len(signal_samples)) - position)
signal[position:position + insert_length] = signal_samples[:insert_length]


# 7)
sigma = float(input("Введите значение отклонения (sigma): "))
noise = np.random.normal(0, sigma, 2 * len(signal_samples))
noisy_signal = [s + n for s, n in zip(signal, noise)]


# 8)
corrected_signal = correlation_receiver(noisy_signal, bits_to_samples(gold_sequence, N), len(signal_samples))

# 9)
bit_sequence_crc_gold_restored = samples_to_bits(corrected_signal, N)

plt.figure(figsize=(13, 10))
plt.step(range(len(bit_sequence_crc_gold_restored)), bit_sequence_crc_gold_restored, where='post', color='b',
         linewidth=2)

# 10)
bit_sequence_crc_restored = bit_sequence_crc_gold_restored[len_sequence:]

# 11)
if check_packet(bit_sequence_crc_restored, G) == True:

    # 12)
    bit_sequence_restored = bit_sequence_crc_restored[:-(len(G) - 1)]
    restored_text = ascii_decoder(bit_sequence_restored)



bit_sequence_crc_gold_stop_4 = []
for a, b in zip(bit_sequence_crc_gold_stop[::2], bit_sequence_crc_gold_stop[1::2]):
    bit_4 = amplitude4(a, b)
    bit_sequence_crc_gold_stop_4.append(bit_4)

gold_sequence_4 = []
for a, b in zip(gold_sequence[::2], gold_sequence[1::2]):
    bit_4 = amplitude4(a, b)
    gold_sequence_4.append(bit_4)

stop_word_4 = []
for a, b in zip(stop_word[::2], stop_word[1::2]):
    bit_4 = amplitude4(a, b)
    stop_word_4.append(bit_4)

signal_samples_4 = bits_to_samples(bit_sequence_crc_gold_stop_4, N)
signal_4 = [0] * (2 * len(signal_samples_4))
insert_length_4 = min(len(signal_samples_4), (2 * len(signal_samples_4)) - position)
signal_4[position:position + insert_length_4] = signal_samples_4[:insert_length_4]

signal_samples = signal_samples[:-160]
signal_samples_4 = signal_samples_4[:-80]


sigma_values = np.linspace(0.1, 1, 19)
ber_values_2 = []
ber_values_4 = []
num_trials = 2

for sigma in sigma_values:
    ber_trials_2 = []
    ber_trials_4 = []
    for _ in range(num_trials):
        noise = np.random.normal(0, sigma, len(signal))
        noisy_signal_2 = [s + n for s, n in zip(signal, noise)]
        corrected_signal_2 = correlation_receiver(noisy_signal_2, bits_to_samples(gold_sequence, N),
                                                  len(signal_samples))
        decode_noisy_signal_2 = decode_noisy_signal_two(corrected_signal_2)
        ber_2 = compute_ber(signal_samples, decode_noisy_signal_2)
        ber_trials_2.append(ber_2)

        noise = np.random.normal(0, sigma, len(signal_4))
        noisy_signal_4 = [s + n for s, n in zip(signal_4, noise)]
        corrected_signal_4 = correlation_receiver(noisy_signal_4, bits_to_samples(gold_sequence_4, N),
                                                  len(signal_samples_4))
        decode_noisy_signal_4 = decode_noisy_signal_four(corrected_signal_4)
        ber_4 = compute_ber(signal_samples_4, decode_noisy_signal_4)
        ber_trials_4.append(ber_4)

    ber_values_2.append(np.mean(ber_trials_2))
    ber_values_4.append(np.mean(ber_trials_4))

plt.figure(figsize=(10, 10))
plt.plot(sigma_values, ber_values_2, marker='o', linestyle='-', color='blue', label='BER2')
plt.plot(sigma_values, ber_values_4, marker='o', linestyle='-', color='orange', label='BER4')
plt.xlabel('Sigma (σ)')
plt.ylabel('BER')
plt.title('Зависимость BER от σ')
plt.grid(True)

# Настройка частоты отображения значений по оси X
plt.xticks(ticks=sigma_values, labels=[f"{sigma:.2f}" for sigma in sigma_values], rotation=45)

plt.legend()
plt.show()
