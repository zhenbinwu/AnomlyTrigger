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

    i = pickle.load(open("./%s.p" % modelname, "rb"))
    features = sum([v[0] for b, v in PhysicsObt.items()])
    DrawLoss(modelname, i, features)
    DrawROC(modelname, i, features)

