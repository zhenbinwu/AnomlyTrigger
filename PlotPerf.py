#!/usr/bin/env python
# encoding: utf-8

# File        : PlotPerf.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2019 Oct 16
#
# Description : 



import pickle
import argparse
from Common import *


if __name__ == "__main__":
    ## Import different model/config
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-m", '--module', default="./PuppiJetModel.py",help='module file')
    args = parser.parse_args()
    modulename = os.path.splitext(os.path.basename(args.module))[0]
    exec("from %s import *" % modulename)

    criterion = torch.nn.L1Loss(reduction='sum')
    features = sum([v[0] for b, v in PhysicsObt.items()])
    model = autoencoder(features)
    model.load_state_dict(torch.load("%s.pth" % modelname, map_location=lambda storage, loc: storage))
    lossMap = {}
    for k, v in sampleMap.items():
        lossMap[k] = EvalLoss(v["file"], PhysicsObt, model, criterion,
                              cut=globalcutfunc, noisemaker=v.get("noisetype", None) )

    # lossMap = pickle.load(open("./%s.p" % modelname, "rb"))
    DrawLoss(modelname, lossMap, features)
    DrawROC(modelname, lossMap, features)



