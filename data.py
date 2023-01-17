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
        if f'{yr} {mm} {dd}' in line and nb != num:
            return num
        if line == '':
            return num
    return 0

def make_data(nr_of_lines, file):
    data = []
    for num, line in enumerate(file, 1):
        if num <= nr_of_lines:
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
            data.append([i for i in result if i != 'B'])
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
    data_file = open("vegasdata.txt", 'r')

    year = int(input('What year do you want? Choose from 1935 - 2022: '))
    while year > 1935 or year < 2022:
        year = int(input('Please enter a valid value! Choose from 1935 - 2022: '))

    month, day = input('Choose the month and day as format mm dd: ').split(' ')

    number = find_line_number(year, month, day, data_file)
    number_of_lines = find_data_line_numbers(number, year, month, day, data_file)
    raw_data = make_data(number_of_lines, data_file)
    print(get_data(raw_data))
