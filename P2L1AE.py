#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import uproot
import numpy as np
from collections import OrderedDict
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pickle
import argparse

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from Common import *
# from PuppiJetModel import *
import time
import imp


if __name__ == "__main__":
    ## Import different model/config
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-m", '--module', default="./PuppiJetModel.py",help='module file')
    args = parser.parse_args()
    modulename = os.path.splitext(os.path.basename(args.module))[0]
    exec("from %s import *" % modulename)
    sys.stdout = open("%s.log" % modelname, 'w')

    start = time.time()
    features = sum([v[0] for b, v in PhysicsObt.items()])

    bg = P2L1NTP(bg_files, PhysicsObt)
    dataloader = DataLoader(bg, batch_size=batch_size, pin_memory=True, num_workers=2, shuffle=False)

    model = autoencoder(features)
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = torch.nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 
                                  mode='min',
                                  factor=0.3,
                                  patience=3,
                                  verbose=1,
                                  threshold=1e-4,
                                  cooldown=2,
                                  min_lr=1e-7
                                 )

    for epoch in range(num_epochs):
        for batch_idx, bg_data in enumerate(dataloader):
            _bg_img = Variable(bg_data.type(torch.FloatTensor))
            if torch.cuda.is_available():
                _bg_img = _bg_img.cuda()
            if batch_idx < (len(dataloader)*trainingfrac):
                # ===================forward=====================
                out = model(_bg_img)
                loss = criterion(out, _bg_img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # =================validation===================
            else:
                val_out = model(_bg_img)
                val_loss = criterion(val_out, _bg_img)
            # ===================log========================
            break
        print(loss.data)
        print('epoch [{}/{}], loss:{:.4f}, val loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data, 0))
        # print('epoch [{}/{}], loss:{:.4f}, val loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data, val_loss.data))
        scheduler.step(loss)

    torch.save(model.state_dict(), './%s.pth' % modelname)
    end=time.time()
    print('Traiing time {} mins'.format((end-start)/60))

    criterion = torch.nn.L1Loss(reduction='none')
    lossMap = {}
    for k, v in sampleMap.items():
        lossMap[k] = EvalLoss(v["file"], PhysicsObt, model, criterion)
    DrawLoss(modelname, lossMap, features)
    pickle.dump(lossMap, open("%s.p" % modelname, "wb"))



