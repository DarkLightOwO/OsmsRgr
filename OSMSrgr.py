import numpy as np
import matplotlib.pyplot as plt

#2
def encode_name_to_bits(name):
    bytes_name = name.encode('ascii')
    bits = ''.join(format(byte, '08b') for byte in bytes_name)
    bit_array = [int(bit) for bit in bits]
    return bit_array

#3
def crc_generate(packet, generator, crc_length):
    crc = packet[:] + [0] * (crc_length)
    for i in range(len(packet)):
        if packet[i] == 1:
            for j in range(crc_length + 1):
                crc[i + j] = crc[i + j] ^ generator[j]
    return crc[-crc_length:]

#4
def gold_generate(lenght):
    x = [0, 1, 0, 1, 1] #11
    y = [1, 0, 0, 1, 0] #18
    gold_sequence = []
    
    for i in range(lenght):
        gold_sequence.append(x[4] ^ y[4])

        temp = x[3] ^ x[4]
        x = [temp] + x[:4]

        temp = y[1] ^ y[4]
        y = [temp] + y[:4]

    return np.array(gold_sequence)

#8
def gold_check(SignalWithNoice, gold):
    gold = np.repeat(gold, 5)
    cor = np.correlate(SignalWithNoice, gold, "valid")
    cor/= cor[cor.argmax()]
    max_arg = np.argmax(cor)
    
    plt.figure(10)
    plt.title('Корреляция')
    plt.xlabel('Бит')
    plt.ylabel('Значение корреляции')
    plt.plot(cor)
    
    return max_arg

#9
def Antinoice(arr):
    avg = []
    for i in range(0, len(arr), 5):
        avg.append(sum(arr[i:i+5])/5) 
    arr = avg
    fin = []
    for i in range(len(arr)):
        if arr[i] > 0.5:
            fin.append(1)
        else:
            fin.append(0)
    return fin

#11
def crc_check(packet, generator, crc_length):
    calculated_crc = crc_generate(packet[:-crc_length], generator, crc_length)
    return calculated_crc == packet[-crc_length:]

#12
def decode_bits_to_name(bit_array):
    bit_string = ''.join(str(bit) for bit in bit_array)
    bytes_list = [bit_string[i:i+8] for i in range(0, len(bit_string), 8)]
    name = ''.join(chr(int(byte, 2)) for byte in bytes_list)
    return name

# 1
name = input ("Введите ваше имя и фамилию: ")

#2
bit_array = encode_name_to_bits(name)
print('Битовая последовательность', bit_array)

plt.figure(1)
plt.title('Битовая последовательность')
plt.xlabel('Биты')
plt.ylabel('Амплитуда')
plt.plot(bit_array)

#3
poly = [1, 0, 1, 1, 1, 0, 1, 1] # Порождающий полином (1, 0, 1, 1, 1, 0, 1, 1) 
CRC_LENGTH = len(poly) - 1
crcfin = crc_generate(bit_array, poly, CRC_LENGTH)
print(f"CRC-генератор: {crcfin}")

#4
goldfin = gold_generate(31)
print(f"Gold-генератор: {goldfin}")
data = np.concatenate((goldfin, bit_array, crcfin))

plt.figure(2)
plt.title('Синхронизация, данные и CRC')
plt.xlabel('Биты')
plt.ylabel('Амплитуда')
plt.plot(data)

#5
data5X = np.repeat(data, 5)

plt.figure(3)
plt.title('Синхронизация, данные и CRC(амплитудная модуляция)')
plt.xlabel('Временые отсчёты')
plt.ylabel('Амплитуда')
plt.plot(data5X)

#6
Signal = np.zeros(len(data5X)*2)
datalen = len(data5X)
pos = int(input (f"Введите (1 - {datalen}): "))
Signal [pos : pos + len(data5X)] = data5X

plt.figure(4)
plt.title('Отправляемый сигнал')
plt.xlabel('Время(1сек)')
plt.ylabel('Амплитуда')
plt.plot(Signal)

#7
noise = np.random.normal(0, 0.2, len(Signal))
Signalnoice = noise + Signal

plt.figure(5)
plt.title('Полученный сигнал и шум')
plt.xlabel('Время(1сек)')
plt.ylabel('Амплитуда')
plt.plot(Signalnoice)

#8
SignalStart = gold_check(Signalnoice , goldfin)
Signalfin = Signalnoice [SignalStart:]

plt.figure(6)
plt.title('Полученный сигнал и шум начиная с синхронизации')
plt.xlabel('Время(<1сек)')
plt.ylabel('Амплитуда')
plt.plot(Signalfin)

#9
Signalfin = Signalfin[:len(data5X)]
Signalclear = Antinoice(Signalfin)

plt.figure(7)
plt.title('Полученные данные начиная с синхронизации и заканчивая CRC, без шума')
plt.xlabel('Время(<1сек)')
plt.ylabel('Амплитуда')
plt.plot(Signalclear)

#10
Signalclear = Signalclear[len(goldfin):]
print('Полученные данные без синхронизации', Signalclear)

#11
if crc_check(Signalclear, poly, CRC_LENGTH):
    print("Предача без ошибок")
else:
    print("ERROR! try again XD")
    quit()

#12
Signalclear = Signalclear[:-CRC_LENGTH]
print('Полученные данные без синхронизации и CRC', Signalclear)
decoded = decode_bits_to_name(Signalclear)
print('Слова после расшифровки:', decoded)

#13
fft_up = np.fft.fft(Signal)
fft_up = abs(np.fft.fftshift(fft_up))
fft_down = np.fft.fft(Signalnoice)
fft_down = abs(np.fft.fftshift(fft_down))+100
x = np.arange(-len(fft_up)/2, len(fft_up)/2)

plt.figure(8)
plt.title('Спектры полученного и переданного сигнала')
plt.xlabel('Частота[Гц]')
plt.ylabel('Амплитуда')
plt.plot(x, fft_up, label = 'Переданный')
plt.plot(x, fft_down, label = 'Полученый')
plt.legend()

data05X = np.repeat(data, 3)
data1X = np.repeat(data, 5)
data1X = data1X[:len(data05X)]
data2X = np.repeat(data, 10)
data2X = data2X[:len(data05X)]

fft05X = np.fft.fft(data05X)
fft05X = abs(np.fft.fftshift(fft05X))
fft1X = np.fft.fft(data1X)
fft1X = abs(np.fft.fftshift(fft1X))+50
fft2X = np.fft.fft(data2X)
fft2X = abs(np.fft.fftshift(fft2X))+100
x = np.arange(-len(fft05X)/2, len(fft05X)/2)

plt.figure(9)
plt.title('Спектры 3 разных по длительности сигналов')
plt.xlabel('Частота[Гц]')
plt.ylabel('Амплитуда')
plt.plot(x, fft05X, label = "0.5X")
plt.plot(x, fft1X, label = "1X")
plt.plot(x, fft2X, label = "2X")
plt.legend()
plt.show()