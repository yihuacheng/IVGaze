import os, sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import model
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, yaml, copy
from easydict import EasyDict as edict
import ctools, gtools
import argparse

def main(train, test):

    # ===============================> Setup <============================
    reader = importlib.import_module("reader." + test.reader)

    data = test.data
    load = test.load
    torch.cuda.set_device(test.device)
  

    # ==============================> Read Data <======================== 
    data.origin, folder = ctools.readfolder(data.origin, [test.person])
    data.norm, folder = ctools.readfolder(data.norm, [test.person])

    testname = folder[test.person] 

    dataset = reader.loader(data, 500, num_workers=4, shuffle=True)

    modelpath = os.path.join(train.save.metapath, 
                                train.save.folder, f'checkpoint/{testname}')
    logpath = os.path.join(train.save.metapath, 
                                train.save.folder, f'{test.savename}/{testname}')

    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # =============================> Test <==============================

    begin = load.begin_step; end = load.end_step; step = load.steps

    for saveiter in range(begin, end+step, step):
        print(f"Test {saveiter}") 

        # ----------------------Load Model------------------------------
        net = model.Model()

        
        statedict = torch.load(
            os.path.join(modelpath, f"Iter_{saveiter}_{train.save.model_name}.pt"),
            map_location={f"cuda:{train.device}":f"cuda:{test.device}"}
        )


        net.cuda(); net.load_state_dict(statedict); net.eval()

        length = len(dataset); accs = 0; count = 0

        # -----------------------Open log file--------------------------------
        logname = f"{saveiter}.log"
        
        outfile1 =  open(os.path.join(logpath, logname + '_zone'), 'w')
        outfile1.write("name results gts\n")

        outfile2 =  open(os.path.join(logpath, logname + '_dir'), 'w')
        outfile2.write("name results gts\n")



        # -------------------------Testing---------------------------------
        with torch.no_grad():
            n_true = {}
            n_total = {}

            for j, (data, label) in enumerate(dataset):

                for key in data:
                    if key != 'name': data[key] = data[key].cuda()

                names =  data["name"]

                gt_zones = label.zone.view(-1)
                gt_dirs = label.originGaze
                gt_dirs = label.normGaze
                gazes, zones, _, _ = net(data, train=False)

                
                for k, cls in enumerate(zones):

                    gt = str(int(gt_zones[k]))
                    name = [names[k]]
                    if gt == str(int(cls)):
                        n_true[gt] = 1 + n_true.get(gt, 0)

                    n_total[gt] = 1 + n_total.get(gt, 0)

                    log = name + [f"{cls}"] + [f"{gt}"]
                    outfile1.write(" ".join(log) + "\n")

                for k, gaze in enumerate(gazes):

                    gaze = gaze.cpu().detach().numpy()
                    gt = gt_dirs.numpy()[k]

                    count += 1
                    accs += gtools.angular(
                                gtools.gazeto3d(gaze), 
                                gtools.gazeto3d(gt)
                            )
            
                    name = [names[k]]
                    gaze = [str(u) for u in gaze] 
                    gt = [str(u) for u in gt] 
                    log = name + [",".join(gaze)] + [",".join(gt)]
                    outfile2.write(" ".join(log) + "\n")

            keys = sorted(list(n_true.keys()), key = lambda x:int(x))
            true_num = 0
            total_num = 0
            for key in keys:
                true_num += n_true[key]
                total_num += n_total[key]
                loger = f'Class {key} {n_true[key]} {n_total[key]} {n_true[key]/n_total[key]:.3f}\n'
                outfile1.write(loger) 
            loger = f"[{saveiter}] Total Num: {total_num}, True: {true_num}, AP:{true_num/total_num:.3f}" 
            outfile1.write(loger)
            print(loger)

            loger = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
            outfile2.write(loger)
            print(loger)
        outfile1.close()
        outfile2.close()

if __name__ == "__main__":


   # Read model from train config and Test data in test config.
    train_conf = edict(yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader))

    test_conf = edict(yaml.load(open(sys.argv[2]), Loader=yaml.FullLoader))

    for i in range(int(sys.argv[3])):
        test_conf_cur = copy.deepcopy(test_conf)

        test_conf_cur.person = i

        print("=======================>(Begin) Config of training<======================")

        print(ctools.DictDumps(train_conf))

        print("=======================>(End) Config of training<======================")

        print("")

        print("=======================>(Begin) Config for test<======================")

        print(ctools.DictDumps(test_conf_cur))

        print("=======================>(End) Config for test<======================")

        main(train_conf.train, test_conf_cur)

