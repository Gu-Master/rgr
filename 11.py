import matplotlib.pyplot as plt
import numpy as np
import random

def string_to_bits(s):
    result = []
    for char in s:
        bits = bin(ord(char))[2:].zfill(8)
        result.extend([int(bit) for bit in bits])
    return result

def calculate_crc(data, generator):
    data_size = len(data)
    generator_size = len(generator)
    extended_data = data + [0] * (generator_size - 1)

    for i in range(data_size):
        if extended_data[i] == 1:
            for j in range(generator_size):
                extended_data[i + j] ^= generator[j]

    crc = extended_data[data_size:]
    return crc

def generate_gold_sequence(reg1_init, reg2_init, seq_length):
    reg1 = reg1_init[:]
    reg2 = reg2_init[:]
    gold_sequence = []

    for _ in range(seq_length):
        out_reg1 = reg1[4]
        out_reg2 = reg2[4]

        feedback1 = reg1[1] ^ reg1[4]
        reg1 = [feedback1] + reg1[:-1]

        feedback2 = reg2[0] ^ reg2[1] ^ reg2[2]
        reg2 = [feedback2] + reg2[:-1]

        gold_sequence.append(out_reg1 ^ out_reg2)

    return gold_sequence

def bits_to_time_samples(bits, samples_per_bit):
    time_samples = []
    for bit in bits:
        time_samples.extend([bit] * samples_per_bit)
    return time_samples

def decode_time_samples(time_samples, samples_per_bit, threshold):
    decoded_bits = []
    for i in range(0, len(time_samples) - (len(time_samples) % samples_per_bit), samples_per_bit):
        chunk = time_samples[i:i + samples_per_bit]
        mean_value = np.mean(chunk)
        decoded_bits.append(1 if mean_value > threshold else 0)
    return decoded_bits

def bits_to_string(bits):
    string = ""
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        char_code = int("".join(str(bit) for bit in byte), 2)
        string += chr(char_code)
    return string

def visualize_spectrum(signal, fs, title="Signal Spectrum"):
    frequencies = np.fft.fftfreq(len(signal), 1/fs)
    spectrum = np.abs(np.fft.fft(signal))

    plt.figure(figsize=(10, 4))
    plt.plot(frequencies, spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    name = "Gurachevskii"
    surname = "Nikita"

    full_name = name + surname

    bit_sequence = string_to_bits(full_name)

    generator = [1, 0, 1, 1, 1, 0, 1, 1]
    crc = calculate_crc(bit_sequence, generator)

    bit_sequence_with_crc = bit_sequence + crc

    reg1_init = [1, 0, 1, 0, 1]
    reg2_init = [1, 1, 1, 0, 1]
    gold_sequence_length = 31
    gold_sequence = generate_gold_sequence(reg1_init, reg2_init, gold_sequence_length)

    final_bit_sequence = gold_sequence + bit_sequence_with_crc

    samples_per_bit = 10
    time_samples = bits_to_time_samples(final_bit_sequence, samples_per_bit)

    noisy_signal = np.array(time_samples) + np.random.normal(0, 0.5, len(time_samples))

    threshold = 0.5
    decoded_bits = decode_time_samples(noisy_signal, samples_per_bit, threshold)

    decoded_bits_without_gold = decoded_bits[len(gold_sequence):]

    decoded_data_bits = decoded_bits_without_gold[:-len(crc)]

    decoded_string = bits_to_string(decoded_data_bits)

    print("Декодировано сообщение по CRC сигналу:", decoded_string)
