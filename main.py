import re

dataset = []


def check_packet(packet):
    packet_data = packet.split(',')

    # Check packet contains both device id and value
    if len(packet_data) != 2:
        print("Packet invalid, does not contain address and value")
        return False

    device_id = packet_data[0]
    if len(device_id) != 16:
        print("Packet invalid, address length is incorrect")
        return False

    try:
        int(device_id, 16)
    except ValueError:
        print("Packet invalid, address is not valid hex")
        return False

    try:
        value = float(packet_data[1])
    except ValueError:
        print("Packet invalid, value is not a decimal")
        return False

    return device_id, value


def load_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            # Attempt to find packet data
            packet_data = re.findall("'(.*?)'", line)

            if len(packet_data) == 1:
                packet_values = check_packet(packet_data[0])

                if packet_values:
                    dataset.append(packet_values)


def calculate_average(limit=None):
    averages = {}

    i = 0

    for data in dataset:
        if limit is not None and i >= limit:
            break

        if data[0] in averages:
            # Add new value to running average
            averages[data[0]] = (averages[data[0]] + data[1]) / 2
        else:
            averages[data[0]] = data[1]

        i += 1

    return averages


if __name__ == '__main__':
    load_dataset("data.txt")
    print(dataset)
    print(calculate_average())
