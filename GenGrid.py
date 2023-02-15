
# Elham Sadeghi February 2023

#This file generates logs of runs with various parameters

import os
import time
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sequenceModel import SequenceModel

from sequenceTrainer import SequenceTrainer
from sequenceDataset import SequenceDataset
from plotRun import genPlotForRun
import matplotlib.pyplot as plt



#CPU or GPU?
os.environ["CUDA_VISIBLE_DEVICES"]=""

batchSize = 32

numEpochs = 500
alphas = [0.003]
betas = [0.007]
gammas = [2]
deltas = [1]
dropout =  [0]
latentDims= [64]
d_model = [4]
num_heads = [2] #it must be devisible by d_model
stack = [1]
dim_feedforward=[32]
trainTestSplit = (0.9, 0.1)
weighted = True



#Post processing of log data, avarages metrics for each epoch
def averageCols(logMat):
    
    print(logMat.shape)
    rv = np.zeros((numEpochs, 6))
    for epoch in range(numEpochs):
        for col in range(1, 6):
            num = 0
            for row in range(logMat.shape[0]):
                if(logMat[row, 0] == epoch):
                    rv[epoch, col] += logMat[row, col]
                    num += 1
            rv[epoch, col] /= num
        rv[epoch,0] = epoch
    
    return rv
    


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

paramDict = {"alpha":alphas, "beta": betas, "gamma": gammas, "delta": deltas, 
     "latentDims": latentDims,  "dropout":dropout, "d_model":d_model, "num_heads":num_heads,'stack':stack,'dim_feedforward':dim_feedforward}
for params in list(ParameterGrid(paramDict)):   #gridsearch
    #set up the model and trainer
    model = SequenceModel(emb_dim=params["latentDims"], dropout=params["dropout"], d_model=params["d_model"], num_heads=params["num_heads"], stack=params["stack"], dim_feedforward=params["dim_feedforward"] ) 
    data = SequenceDataset(datafile = './data-and-cleaning/cleandata_4Feb.csv', split=trainTestSplit)
    
    if torch.cuda.is_available(): 
        print('cuda available')
        model.cuda()
    trainer = SequenceTrainer(data, model, alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"], delta=params["delta"], logTerms=True, IICorVsEpoch=True)
    if torch.cuda.is_available(): 
        trainer.cuda()
    
    #train model, use internal logging
    print("Training Model")
    # trainer.train_model(batchSize, numEpochs, log=False)

    #train model, using weighted loss function
    wavg,wavgv , liiavg , liiavgv = trainer.train_model(batchSize, numEpochs, log=False, weightedLoss=weighted)

    filename = "a" + str(params["alpha"]) + "lds"+str(params["latentDims"])+"b"+str(params["beta"])+"g" +str(params["gamma"])+"d"+str(params["delta"])+"DM"+str(params["d_model"])+"NH"+str(params["num_heads"])+'st'+str(params["stack"])+'dff'+str(params["dim_feedforward"])
    #save the model
    if weighted:
        torch.save(model, "./models/weighted/" + filename + ".pt")
    else:
        torch.save(model, "./models/" + filename + ".pt")
    print("Avging stats for batches")
    #training accuricies at each epoch
    tl = averageCols(trainer.trainList)
    #validation accuricies at each epoch
    vl = averageCols(trainer.validList)

    print("Computing final correlations")
    #Corellation on II at every 50 epochs, if it exists
    wldims = trainer.WLCorList if trainer.IICorVsEpoch else np.zeros((0, model.emb_dim))
    liidims = trainer.LIICorList if trainer.IICorVsEpoch else np.zeros((0, model.emb_dim))

    wldimsv = trainer.WLCorListv if trainer.IICorVsEpoch else np.zeros((0, model.emb_dim))
    liidimsv = trainer.LIICorListv if trainer.IICorVsEpoch else np.zeros((0, model.emb_dim))

    wtldims = trainer.WLtauList if trainer.IICorVsEpoch else np.zeros((0, model.emb_dim))
    liitdims = trainer.LIItauList if trainer.IICorVsEpoch else np.zeros((0, model.emb_dim))

    wtldimsv = trainer.WLtauListv if trainer.IICorVsEpoch else np.zeros((0, model.emb_dim))
    liitdimsv = trainer.LIItauListv if trainer.IICorVsEpoch else np.zeros((0, model.emb_dim))
    
    print("Saving file")
    if os.path.isdir('./runs/weighted/') == False:
        os.mkdir('./runs/weighted/')
    par = np.array([params["alpha"], params["beta"], params["gamma"], params["delta"], 
        params["latentDims"], params["dropout"], params["d_model"], params["num_heads"], params["stack"], params["dim_feedforward"]])
    if weighted:
        np.savez("./runs/weighted/" + filename + ".npz", par=par, tl=tl, vl=vl, wldims=wldims, liidims=liidims ,wldimsv=wldimsv, liidimsv=liidimsv ,wtldims=wtldims ,liitdims=liitdims,wtldimsv=wtldimsv ,liitdimsv=liitdimsv,wavg=wavg,wavgv=wavgv,liiavg=liiavg,liiavgv=liiavgv)
    else:
        np.savez("./runs/" + filename + ".npz", par=par, tl=tl, vl=vl, wldims=wldims, liidims=liidims,wldimsv=wldimsv, liidimsv=liidimsv,liiavg=liiavg,liiavgv=liiavgv)
    if weighted:
        genPlotForRun(runsPath="./runs/weighted/", run=filename + ".npz", graphsPath="./graphs/weighted", graph=filename + ".png")
    else:
        genPlotForRun(runsPath="./runs/", run=filename + ".npz", graphsPath="./graphs", graph=filename + ".png")

