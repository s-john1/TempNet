import re
import statistics

dataset = [] # A list of the each individual packet, ordered by time
debug = False

count = [0, 0]

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
                    add_value(packet_values[0], packet_values[1])


def filter_dataset(limit=None):
    filtered = {}

    # Loop through dataset starting with last entries
    for data in reversed(dataset):
        id = data[0]
        value = data[1]

        if id in filtered:
            # Stop counting if a set limit is reached for the device
            if limit is not None and len(filtered[id]) >= limit:
                continue
        else:
            filtered[id] = []

        # Add new value to dataset
        filtered[id].append(value)

    return filtered


def add_value(device, value):
    # We need an appropriate sample size before being able to accurately accept/reject packets
    # so, automatically accept the first x packets
    if len(dataset) <= 10:
        dataset.append((device, value))
        return


    # Fetch last x amount of accepted values for each device
    filtered = filter_dataset(3)
    raw_values = get_raw_values(filtered)

    print(device, value)
    print(raw_values)

    # Get the average value across every sensor in the filtered dataset
    average = calculate_average(raw_values)
    print("Average:", average)

    std_dev = statistics.stdev(raw_values)
    print("Standard Dev:", std_dev)

    if value < average - std_dev*2 or value > average + std_dev*2:
        # Reject if not witin x standard deviations
        print("Rejecting value", value, "from device", device)
        count[0] += 1  # Add to count
    else:
        print("Accepting value", value, "from device", device)
        dataset.append((device, value))
        count[1] += 1  # Add to count
    print()
    if debug:
        output_info()


def get_raw_values(data):
    # Will retrieve the raw values from a dataset

    values = []
    for device in data:
        values += data[device]

    return values


def output_info():
    print(dataset)


def calculate_average(data):
    # Don't run if dataset is empty
    if not data:
        return None

    total = 0
    i = 0

    for value in data:
        total += value
        i += 1

    return total / i


if __name__ == '__main__':
    load_dataset("example_data_4.txt")
    #load_dataset("data.txt")
    print("Rejected:", count[0])
    print("Accepted:", count[1])

