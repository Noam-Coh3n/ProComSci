#
# Name: Lars Bijvoet, Erik Leonards, Noam Cohen & Jelle Sipkes
# Study: Dubbel Bachelor mathematics and Informatica.
#
# data.py:
# Read data from file and store it.

def find_line_number(yr, mm, dd, file):
    for num, line in enumerate(file, 1):
        if f'{yr} {mm} {dd}' in line:
            return num
    return 0

def find_data_line_numbers(nb, yr, mm, dd, file):
    for num, line in enumerate(file, 1):
        if f'{yr} {mm} {dd}' in line and nb == num:
            pass
        if f'{30}   {100}' in line and num > nb:
            return num
        if line == '':
            return num
    return 0

def make_data(nr_of_lines, nr, file):
    data = []
    for num, line in enumerate(file, 1):
        if num == nr:
            continue
        if num <= nr + nr_of_lines - 1:
            print(line)
            result = ''
            for count, ch in enumerate(line, 1):
                if count == 1:
                    result = result + ch + ' '
                if count == 16 or \
                   count == 22 or count == 28:
                    result = result + ' ' + ch
                else:
                    result += ch
            result = list(filter(None, ((result.strip()).split(" "))))
            result = [i for i in result if i != 'B']
            data.append([i for i in result if i != 'A'])
    return data

def get_data(raw_data):
    height, press, temp, wdir, wspd = 1, 0, 2, 3, 4
    order = [height, press, temp, wdir, wspd]
    data = []
    for i in raw_data:
        data1 = []
        for k, l in enumerate(i, 1):
            if k == 4 or k == 5 or k == 6 or k == 9 or k == 10:
                data1.append(int(l))
        data1 = [data1[j] for j in order]
        data.append(data1)
    return data

if __name__ == "__main__":
    year = int(input('What year do you want? Choose from 2018 - 2023: '))
    while year < 2018 or year > 2023:
        year = int(input('Please enter a valid value! Choose from 2018 - 2023: '))

    month, day = input('Choose the month and day as format mm dd: ').split(' ')
    with open("vegasdata.txt", 'r') as data_file:

        number = find_line_number(year, month, day, data_file)
        print(number)
        number_of_lines = find_data_line_numbers(number, year, month, day, data_file)
        data_file.seek(0)
        data_file = data_file.readlines()
        raw_data = make_data(number_of_lines, number, data_file)
        print(get_data(raw_data))
