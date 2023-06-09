import time
# import logging
import sys
import os
import numpy as np

#_logger = logging.getLogger("cifar10_resnet50_nomad")

#logging.basicConfig(level=logging.DEBUG,
#                    format='(%(threadName)-9s) %(message)s',)

# 4GPU
gpuNumber = "4,5,6,7"
evalScript = "mpirun -np 4 python3 eval.py"

singleOutputBB ='outputBB.txt'
logAllFile = 'logAllOutputs.txt'

dataset_path='/local1/datasets/imagenet/ILSVRC2012'
epoch_size = 30

previousHistoryFileIsProvided = False 
previousHistoryFileName = "history.prev.txt"

# This function must be customized to the problem (order and value of the variables)
def get_parameters_from_Nomad_input(x):
    
    config = dict()
    # print(x)
    
    
#1    "batch_size": {"_type":"choice", "_value": [32, 64, 128, 256, 512]},
#2    "weight_decay": {"_type":"choice", "_value": [0, 0.00000001,  0.0000001, 0.000001]},
#3    "lr_min":{"_type":"choice", "uniform":[-6, -3]},
#4    "lr_max": {"_type":"choice", "uniform":[-1, -3]},
#5    "optimizer":{"_type":"choice", "_value":["SGD", ,"Momentum", "Adagrad", "Adam", "Adamax"]}
    
    # 1 -> batch size
    if x[0] == 1:
        valueBS = 32
    elif x[0] == 2:
        valueBS = 64
    elif x[0] == 3:
        valueBS = 128
    elif x[0] == 4:
        valueBS = 256
    elif x[0] == 5:
        valueBS = 512
    else:
        return config
    config['batch_size'] = valueBS
     
    # 2 -> weight decay
    if x[1] == 1:
        valueWD = 0
    elif x[1] == 2:
        valueWD = 0.00000001
    elif x[1] == 3:
        valueWD = 0.0000001
    elif x[1] == 4:
        valueWD = 0.000001
    else:
        return config
    config['weight_decay'] = valueWD
    
    
    # !!!!!
    # 3 -> Min learning rate: min_lr = 10^x3
    # !!!!!
    if x[2] < 0:
        valueMinLR = pow(10,x[2])
    else:
        return config
    # print(config)
    config['min_lr'] = valueMinLR

    # !!!!!
    # 4 -> Max learning rate: max_lr = 10^x4
    # !!!!!
    if x[3] < 0:
        valueMaxLR = pow(10,x[3])
    else:
        return config
    # print(config)
    config['max_lr'] = valueMaxLR
 
    # 5 -> Optimizer
    if x[4] == 1:
        valueOptim = 'SGD'
    elif x[4] == 2:
        valueOptim = 'Momentum'
    elif x[4] == 3:
        valueOptim = 'Adagrad'
    elif x[4] == 4:
        valueOptim = 'Adam'
    elif x[4] == 5:
        valueOptim = 'Adamax'
    else:
        return config
    config['optimizer'] = valueOptim

    return config



if __name__ == '__main__':

    inputFileName=sys.argv[1]
    X=np.fromfile(inputFileName,sep=" ")

    if previousHistoryFileIsProvided:
        dim = len(X)
        # Read a history file from a run
        rawEvals = np.fromfile(previousHistoryFileName,sep=" ")
        # Each line contains 2 values for X and a single output (objective value)
        nbRows = rawEvals.size/(len(X)+1)
        # print(nbRows)
        # Split the whole array is subarrays
        npEvals = np.split(rawEvals,nbRows)
        for oneEval in npEvals:
            # print('Dim=',dim,' ',oneEval)
            diffX=X-oneEval[0:dim]
            # print(np.linalg.norm(diffX))
            if np.linalg.norm(diffX) < 1E-10:
                fout=oneEval[dim]
                print(fout)
                # print('Find point in file: ',X,' f(x)=',fout)
                exit()

    # Interpret input to args passed to eval script
    #1    "batch_size": {"_type":"choice", "_value": [32, 64, 128, 256, 512]},
    #2    "weight_decay": {"_type":"choice", "_value": [0, 0.00000001,  0.0000001, 0.000001]},
    #3    "min_lr":{"_type":"choice", "uniform":[-6, -3]},
    #4    "max_lr": {"_type":"choice", "uniform":[-1, -3]},
    #5    "optimizer":{"_type":"choice", "_value":["SGD", ,"Momentum", "Adarad", "Adam", "Adamax"]}
    args = get_parameters_from_Nomad_input(X)
    # print(RCV_CONFIG)
    #_logger.debug(args)

    
    syst_cmd = 'singularity exec --nv -B /local1/ctribes/mindspore -B /local1/datasets /local1/ctribes/singularity_docker_mindspore/mindspore-gpu-cuda11.6_2.0.0-alpha.sif '
    syst_cmd += ' bash -c \'' 
    syst_cmd += ' CUDA_VISIBLE_DEVICES=' +gpuNumber + ' ' + evalScript + ' --optimizer=' + args['optimizer']  \
                                                                     + ' --batch_size=' + str(args['batch_size'])  \
                                                                     + ' --weight_decay=' + str(args['weight_decay']) \
                                                                     + ' --min_lr=' + str(args['min_lr']) \
                                                                     + ' --max_lr=' + str(args['max_lr']) \
                                                                     + ' --batch_size=' + str(args['batch_size']) \
                                                                     + ' --epoch_size=' + str(epoch_size) \
                                                                     + ' --do_train --do_eval --num_classes=1000' \
                                                                     + ' --dataset_path=' + dataset_path \
                                                                     + ' --run_distribute' \
                                                                     + ' > ' + singleOutputBB
    syst_cmd += '\''
    
    
    
    # print(syst_cmd)
    os.system(syst_cmd)
    

    # Default value
    bbOutput='Inf'
    with open(singleOutputBB, "r") as file:
        lines = file.readlines()
        lastLine = lines[len(lines)-1]
        output = lastLine.split("=")
        if output[0] == "Eval_out":
            bbOutput = output[1]
        file.close()
        # append file into log file
        #append_cmd  =  'cat ' + inputFileName + ' >> ' + logAllFile + ' ; '
        #append_cmd +=  'cat ' + singleOutputBB + ' >> ' + logAllFile
        #os.system(syst_cmd)

        print(bbOutput, end=" ")
        

