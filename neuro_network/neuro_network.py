"""
    @author: Suiyi Liu 
    @email: malygosa@mail.ustc.edu.cn
    @brief: A multi-layer neural network from scratch for discrete classification
    We train a multi-layer fully-connected neural network from scratch to classify
    the dataset . An L2 loss  function, sigmoid activation, and no bias terms are assumed.
    The weight optimization is gradient descent via the delta rule.

    All the libraries we utilized are in listed below:
    numpy, os, sys, sklearn.metrics, pickle, matplotlib.pyplot,
    json, math, random, csv, collections
"""

__all__=["NN_model"]

import numpy as np
import os,sys

os.chdir(sys.path[0])

from neuro_network.src.NN import NN
import neuro_network.src.utils as utils
import neuro_network.src.misc as misc
from sklearn.metrics import f1_score,roc_auc_score

def NN_model(csv_filename:str,target_label:str,hidden_layers:list[int],normalize:bool=False,n_folds:int=5,epoch:int=300,
             eta:float=0.01,model_path:str="./model.dat",crossValid:bool=False)->None:
    
    X, y, n_classes = utils.read_csv(csv_filename, target_name=target_label, normalize=normalize)
    N, d = X.shape

    #weight_init=None
    seed_crossval = 1 # seed for cross-validation
    seed_weights = 1 # seed for NN weight initialization

    idx_all = np.arange(0, N)
    idx_folds = utils.crossval_folds(N, n_folds, seed=seed_crossval) # list of list of fold indices

    # Train/evaluate the model on each fold
    for i, idx_valid in enumerate(idx_folds):

        # Collect training and test data from folds
        idx_train = np.delete(idx_all, idx_valid)
        X_train, y_train = X[idx_train], y[idx_train]
        X_valid, y_valid = X[idx_valid], y[idx_valid]

        # Build neural network classifier model and train
        model = NN(input_dim=d, output_dim=n_classes,
                hidden_layers=hidden_layers, seed=seed_weights,
                )
        print("training start, train dataset size:{:d}".format(len(X_train)))
        acc_list,loss_list,_=model.train(X_train, y_train, eta=eta, n_epochs=epoch)

        # Print cross-validation result  
        ypred_valid = model.predict(X_valid) 
        acc=acc_list[-1][-1]
        f1_s=f1_score(y_valid,ypred_valid,average="macro")
        auc=roc_auc_score(y_valid,ypred_valid)
        print("training completed")
        print("ACC:{:.4f} F1-score:{:.4f} AUC:{:.4f}".format(acc,f1_s,auc))
        
        if crossValid==False:
            misc.plot_learning_curve(loss_list)
            misc.save_model(model,model_path) #type:ignore
            return

    

#os.chdir(sys.path[0])
#NN_model("../lab3-data-select.csv","REPEAT",[9],normalize=True,n_folds=5,epoch=100,eta=0.1,model_path="./test.dat")
