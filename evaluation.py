import sys
import os

folder = sys.argv[1]
begin = 10
step = 10
end = 160

# pre_add = 'log_zone'
pre_add = 'log_dir'


def parse(line):
    line = line.strip().split(" ")
    number = int((line[3][:-1]))
    avg = float(line[-1])
    return number, avg * number, avg

def printResult(n_num, n_error, n_avg):
    print(f'Iter {i}: Total: {n_num} Error: {n_error/n_num:.2f}\tSub_Error: ', end = '')
    for avg in n_avg:
        print(f'{avg} ', end='')
    print('')


sub_folders = os.listdir(folder)

for i in range(begin, end+1, step):
    try:
        name = f"{i}.{pre_add}"
        n_num = 0
        n_error = 0
        n_avg = []
        for sub_path in sub_folders:
            filename = os.path.join(folder, sub_path, name)
            with open(filename) as infile:
                lines = infile.readlines()
                number, error, avg = parse(lines[-1])
                n_num += number
                n_error += error
                n_avg.append(avg)

        printResult(n_num, n_error, n_avg)
    except:
        pass
        
            
                
            
