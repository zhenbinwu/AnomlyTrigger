#!/usr/bin/env python
# coding: utf-8

import os
import torch
import socket
import uproot
import glob
import torchvision
import numpy as np
import  matplotlib 
matplotlib.use("Agg")
from collections import OrderedDict
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle

from torch.utils.data import Dataset, DataLoader
print(uproot.__version__) # Need latest uproot v3.7.1 for LazzyArrays


# ## A class for load in data from ROOT file, using uproot
# 
# It should be generic for all kind of flattree
# LazzyArrays is very new for uproot. Need more testing for performances

hostname = socket.gethostname()
folder = "/home/benwu/Data/Phaes2L1Ntuple/"
if hostname == "macbook":
    folder = "/Users/benwu/Data/Phaes2L1Ntuple/"
elif "fnal.gov" in hostname:
    folder = "/uscms_data/d2/lpctrig/benwu/AutoEncoderSample/Phaes2L1Ntuple/"

bg_files  = "%s/NeutrinoGun_E_10GeV_V7_5_2_MERGED.root" % folder
sg_files  = "%s/VBF_HToInvisible_M125_14TeV_pythia8_PU200_V7_4_2.root" % folder
sg_files2 = "%s/VBFHToBB_M-125_14TeV_powheg_pythia8_weightfix_V_7_5_2.root" % folder
sg_files3 = "%s/GluGluToHHTo4B_node_SM_14TeV-madgraph_V7_5_2.root" % folder

sampleMap   = {
    "BG"   :
    {
        "file" : bg_files,
        "histtype" : 'bar',
        "label" : 'BG', 
        "color" : 'yellow',
    },
    # "BG_JetMET"   :  
    # {
        # "file" : bg_files,
        # "histtype" : 'step', 
        # "label" : 'Smeared Jet/MET 1$\sigma$', 
        # "color" : 'b', 
        # "noisetype" : "JetMET50"
    # },
    # "BG_JetEt"   :  
    # {
        # "file" : bg_files,
        # "histtype" : 'step', 
        # "label" : r'Smeared Jet Et 1$\sigma$', 
        # "color" : 'm', 
        # "noisetype" : "PuppiJetEtGau1"
    # },
    # "BG_MET10"   :  
    # {
        # "file" : bg_files,
        # "histtype" : 'step', 
        # "label" : 'MET * 1.1', 
        # "color" : 'r', 
        # "noisetype" : "MET10"
    # },
    # "BG_MET20"   :  
    # {
        # "file" : bg_files,
        # "histtype" : 'step', 
        # "label" : 'MET * 1.2',
        # "color" : 'g', 
        # "noisetype" : "MET20"
    # },
    # "BG_MET50"   :  
    # {
        # "file" : bg_files,
        # "histtype" : 'step', 
        # "label" : 'MET * 1.5',
        # "color" : 'c', 
        # "noisetype" : "MET50"
    # },
    'HtoInvisible' :
    {
        "file" :  sg_files,
        "histtype" : 'step', 
        "label" : 'HtoInvisible', 
        "color" : 'r', 
    },
    'VBFHToBB' : 
    {
        "file" :  sg_files2,
        "histtype" : 'step', 
        "label" : 'VBFHToBB', 
        "color" : 'g', 
    },
    'GluGlutoHHto4B' : 
    {
        "file" :  sg_files3,
        "histtype" : 'step', 
        "label" : 'GluGlutoHHto4B', 
        "color" : 'b', 
    },
}

batch_size = 2000 #144
num_epochs = 100
learning_rate = 1e-3
trainingfrac = 0.8
globalcutfunc = None

class P2L1NTP(Dataset):
    def __init__(self, dir_name, features = None,
                 tree_name="l1PhaseIITree/L1PhaseIITree",
                 sequence_length=50, verbose=False,
                 cutfunc =None, noisetype=None):
        self.tree_name = tree_name
        self.features = features
        self.sequence_length = sequence_length
        self.file_names = glob.glob(dir_name)
        ## Cache will be needed in case we train with >1 eposh
        ## Having issue and reported in https://github.com/scikit-hep/uproot/issues/296
        self.cache = uproot.cache.ArrayCache(1024**3)
        self.upTree = uproot.lazyarrays(self.file_names, self.tree_name, self.features.keys(), cache=self.cache)
        self.noisetype = noisetype
        self.cutfunc = cutfunc
        if self.cutfunc is not None:
            self.upTree = self.upTree[self.cutfunc(self.upTree)]

    def __len__(self):
        if self.cutfunc is None:
            return uproot.numentries(self.file_names, self.tree_name, total=True)
        else:
            return len(self.upTree)

    def __getitem__(self, idx):
        reflatnp = []
        event = self.upTree[idx]
        for b, v in self.features.items():
            g  = event[b]
            ln = v[0]
            scale = v[1]
            if isinstance(g,(int, float)):
                tg = np.array([g])
            else:
                if len(g)>= ln:
                    tg = g[:ln]
                else:
                    tg = np.pad(g, (0, ln-len(g)), 'constant', constant_values=0)
            self.MakingNoise(b, tg)

            if scale > 10 :
                tg = tg / scale
            elif scale > 1 :
                tg = tg + scale
            reflatnp.append(tg)
        org = np.concatenate(reflatnp, axis=0)
        return org

    def MakingNoise(self, varname, var):
        if self.noisetype is None:
            return True

        if self.noisetype == "MET10":
            if varname == "puppiMETEt":
                var = var * 1.1
        if self.noisetype == "MET20":
            if varname == "puppiMETEt":
                var = var * 1.2
        if self.noisetype == "MET50":
            if varname == "puppiMETEt":
                var = var * 1.5
        if self.noisetype == "PuppiJetEtGau1":
            if varname == "puppiJetEt":
                with np.nditer(var, op_flags=['readwrite']) as it:
                    for x in it:
                        x[...] = abs(np.random.normal(x, 1))
        if self.noisetype == "JetMET50":
            if any([varname == x for x in ["puppiJetEt", "puppiJetEta", "puppiJetPhi", "puppiMETPhi"]]):
                with np.nditer(var, op_flags=['readwrite']) as it:
                    for x in it:
                        x[...] = abs(np.random.normal(x, 1))
            if varname == "puppiMETEt":
                var = var * 1.5

    def GetCutArray(self, cutfunc):
        select = cutfunc(self.upTree)
        sel = np.array(select)[np.newaxis]
        nfeatures = sum([v[0] for b, v in self.features.items()])
        ones = np.ones((self.__len__(), nfeatures))
        out = np.multiply(ones, sel.T)
        return out

def EvalLoss(samplefile, PhysicsObt, model, criterion, cut=None, noisemaker=None):
    sample = P2L1NTP(samplefile, PhysicsObt, noisetype=noisemaker)
    dataloader = DataLoader(sample, batch_size=batch_size, pin_memory=True, num_workers=2, shuffle=False)
    for batch_idx, vbg_data in enumerate(dataloader):
        _vbg_img = Variable(vbg_data.type(torch.FloatTensor))
        if torch.cuda.is_available():
            _vbg_img = _vbg_img.cuda()

        vout = model(_vbg_img)
        vloss = criterion(vout, _vbg_img)
        _vbg_out = vout.cpu().detach().numpy()
        _vbg_loss = vloss.cpu().detach().numpy()
        _vbg_data = vbg_data.cpu().detach().numpy()
        if batch_idx == 0:
            vbg_in = _vbg_data
            vbg_out = _vbg_out
            vbg_loss = _vbg_loss
        else:
            vbg_loss = np.append([vbg_loss],[_vbg_loss])
            vbg_out = np.concatenate((vbg_out,_vbg_out))
            vbg_in = np.concatenate((vbg_in, _vbg_data))

    if cut is not None:
        cutmask = sample.GetCutArray(cut)
        vbg_loss = np.multiply(vbg_loss, cutmask.flatten())
        vbg_out = np.multiply(vbg_out, cutmask)

    return vbg_loss, vbg_in, vbg_out

def DrawLoss(modelname, lossMap, features):
    plt.figure(figsize=(8,6))
    bins = np.linspace(0, 30, 60)
    for k, v in lossMap.items():
        reshape_vbg_loss = np.reshape(v, (-1,features))
        vloss = np.sum(reshape_vbg_loss, axis=1).flatten()
        plt.hist(vloss,bins,label=sampleMap[k]['label'],  
                 histtype=sampleMap[k]['histtype'],  
                 color=sampleMap[k]['color'],  normed=True)
    plt.legend(loc='best',fontsize=16)
    plt.xlim(-1,30)
    plt.xlabel('Reconstruction Loss', fontsize=16)
    plt.savefig("%s_Loss.png" % modelname)
    plt.yscale("log")
    plt.savefig("%s_LogLoss.png" % modelname)


def DrawROC(modelname, lossMap, features):
    plt.figure(figsize=(8,6))
    reshape_bg_loss = np.reshape(lossMap["BG"], (-1,features))
    bloss = np.sum(reshape_bg_loss, axis=1).flatten()
    for k, v in lossMap.items():
        if k == "BG":
            continue
        reshape_vbg_loss = np.reshape(v, (-1,features))
        vloss = np.sum(reshape_vbg_loss, axis=1).flatten()
        Tr = np.concatenate(( np.zeros_like(bloss), np.ones_like(vloss)), axis=0)
        Loss = np.concatenate((bloss, vloss), axis=0)
        fpr, tpr, thresholds = roc_curve(Tr, Loss)
        roc_auc = auc(fpr, tpr)
        rate = fpr * 40*1000
        plt.plot(rate, tpr, color=sampleMap[k]['color'],
                 lw=2, label='%s (AUC = %0.2f)' % (sampleMap[k]['label'], roc_auc))
    plt.legend(loc='best',fontsize=16)
    plt.xlabel('Rate (kHz)', fontsize=16)
    plt.ylabel('Signal Efficiency', fontsize=16)
    plt.savefig("%s_ROC.png" % modelname)
    plt.xlim(0,300)
    plt.ylim(0,0.6)
    plt.grid(True)
    plt.savefig("%s_ROCZoom.png" % modelname)


def DrawInOut(modelname, PhysicsObt, inputMap, outputMap):
    for k in inputMap.keys():
        inputData = inputMap[k]
        outputData = outputMap[k]
        for i in range(outputData.shape[1]):
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.hist([inputData[:,i].flatten(), outputData[:,i].flatten()],bins=40, label=['Input', 'Output'])
            plt.yscale('log')
            plt.legend(loc='best')
            plt.subplot(1,2,2)
            plt.hist2d(inputData[:,i].flatten(),outputData[:,i].flatten(),bins=40)
            plt.xlabel('Input')
            plt.ylabel('Output')
            plt.savefig( "%s_%s_%d.png" % (modelname, k, i) )

