import sys
import os
import numpy as np

folder = sys.argv[1]
begin = 20
step = 20
end = 100

pre_add = 'log_zone'

class classAdapt:
    def __init__(self):
        with open ('/home/mercury01/ssd/dataset/Gaze/FaceBased/IVGaze/CVPR_process/class_large.label') as infile:
            lines = infile.readlines()
        self.class_dict = {}
        for line in lines:
            line = line.strip().split(' ')
            number = line[0]

            classname = line[2]
            self.class_dict[number] = classname

    def get(self, name):
        return self.class_dict.get(name, '-1')
        

adapt = classAdapt()


def printMat(matrix):
    size = matrix.shape

    print('pr\gt\t', end = '')
    for i in range(size[1]):
        print(f'{i}', end = '\t')
    print('')

    for i in range(size[0]):
        print(f'{i}', end = '\t')
        for j in range(size[1]):
            print(f'{int(matrix[i, j])}', end = '\t')
        print('')
    return 0


def getAnalysis(results):
    # result: a list of prediction. [[pred, gt], [pred, gt]]
    num = 10
    all_class = range(0, num)

    matrix = np.zeros((num, num))

    total = 0
    for result in results:
        matrix[int(result[1]), int(result[0])] += 1
        total += 1

    printMat(matrix)

    for i in range(num):
        sum_i = np.sum(matrix[i, :])
        acc = matrix[i, i] / sum_i
        print(f'Class {i}: {matrix[i, i]}/{sum_i} = {acc:.3f}')

    return matrix
            

def parse(line):
    line = line.strip().split(" ")
    pred = line[1] 
    gt = line[2] 

    new_pred = adapt.get(pred)
    new_gt = adapt.get(gt)
  
    return line[0], new_pred, new_gt, int(new_pred == new_gt)

sub_folders = os.listdir(folder)

for i in range(begin, end+1, step):

    try:
        name = f"{i}.{pre_add}"

        total = 0
        count = 0
        w_total = 0
        w_count = 0
        n_result = []
        for sub_path in sub_folders:
            
            
            filename = os.path.join(folder, sub_path, name)

            with open(filename) as infile:
                lines = infile.readlines()
                lines.pop(0)

                for line in lines:
                    if len(line.strip().split(' ')) != 3:
                        continue
                    _, pred, gt, result = parse(line)
                    if gt != str(0):
                        total += result
                        count += 1
                    w_total += result
                    w_count += 1
                    n_result.append([pred, gt])
        print(f'***************************Result---{i}*****************************')
        getAnalysis(n_result)
        print(f'Avg: {total}/{count} = {total/count:.3f} ')
        print(f'With 0: {w_total}/{w_count} = {w_total/w_count:.3f} ')
    except:
        pass

        
            
            
