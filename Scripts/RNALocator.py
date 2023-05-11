import datetime
import itertools
from collections import OrderedDict
import argparse
import os
import sys
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import scipy
import statistics
import sys
#import Biopython
import h5py
from sklearn.decomposition import PCA
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf

gpu_options = tf.compat.v1.GPUOptions()
gpu_options.allow_growth = True
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
#sess = tf.compat.v1.Session(config=config)
from tensorflow.keras import backend 
#from keras.backend.tensorflow_backend import set_session
#from tensorflow.keras.backend import set_session
#from keras import backend as K
#tf.compat.v1.keras.backend.set_session(session=sess)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
tf.compat.v1.keras.backend.set_session(sess)
#set_session(session=sess)

from Models.neural_network_predictor import * 
from transcript_info import Gene_Wrapper
from tensorflow.keras.preprocessing.sequence import pad_sequences               
#from tensorflow.keras import backend as K
from tensorflow.python.keras import backend as k
from tensorflow.python.keras import models
from tensorflow.python.keras.models import load_model

gene_ids = None
temp = []
gene_ann = []

batch_size = 512
nb_classes = 4


def label_dist(dist):
    assert (len(dist) == 4)
    return np.array(dist) / np.sum(dist)


def preprocess_data(lower_bound, upper_bound, use_annotations, dataset, max_len):
    
    ''' import data CEFRA-SEQ: CDNA_SCREENED.FA using GENE_WRAPPER calss fromtranascript_gene_data
    '''
    if(dataset == "rnalocate"):
        import rnalocate.processData as processor
        X, y = processor.process_data()
    if(dataset == "cefra-seq"):
        gene_data = Gene_Wrapper.seq_data_loader(use_annotations, dataset, lower_bound, upper_bound)
        
        X= [gene.seq for gene in gene_data]
        y = np.array([label_dist(gene.dist) for gene in gene_data])
        #gene_info = [gene.id for gene in gene_data]

    print("Shape of X", np.shape(X))
    print("Shape of y", np.shape(y))
    from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
    from time import sleep
    
    
    #print("length of gene ids", np.shape(gene_info))


    
    return X, y



def get_kmer(string):
    
    allLexicographic(string)
    
    
    
    # Python program to print all permutations with repetition 
# of characters 

def toString(List): 
    return ''.join(List) 

# The main function that recursively prints all repeated 
# permutations of the given string. It uses data[] to store 
# all permutations one by one 
def allLexicographicRecur (string, data, last, index): 
    
    length = len(string)
    
    
	# One by one fix all characters at the given index and 
	# recur for the subsequent indexes 
    for i in range(length): 
        

        data[index] = string[i] 

        if index==last:
            #print(toString(data))
            
            global temp
            temp.append(toString(data))

        else: 
            allLexicographicRecur(string, data, last, index+1) 
            
    
def allLexicographic(string): 
	length = len(string) 

	# Create a temp array that will be used by 
	# allLexicographicRecur() 
	data = [""] * (length+2)

	# Sort the input string so that we get all output strings in 
	# lexicographically sorted order 
	string = sorted(string) 

	# Now print all permutaions 
	allLexicographicRecur(string, data, length+2, 0)
    
    

    
#allLexicographic(string) 


def printglobal():
    global temp
    print("shape of 5-mer",temp[1])

def findSubsequenceCount(S, T): 
  
    m = len(T) 
    n = len(S) 
  
    # T can't appear as a subsequence in S 
    if m > n: 
        return 0
  
    # mat[i][j] stores the count of  
    # occurrences of T(1..i) in S(1..j). 
    mat = [[0 for _ in range(n + 1)] 
              for __ in range(m + 1)] 
  
    # Initializing first column with all 0s. x 
    # An empty string can't have another 
    # string as suhsequence 
    for i in range(1, m + 1): 
        mat[i][0] = 0
  
    # Initializing first row with all 1s.  
    # An empty string is subsequence of all. 
    for j in range(n + 1): 
        mat[0][j] = 1
  
    # Fill mat[][] in bottom up manner 
    for i in range(1, m + 1): 
        for j in range(1, n + 1): 
  
            # If last characters don't match,  
            # then value is same as the value  
            # without last character in S. 
            if T[i - 1] != S[j - 1]: 
                mat[i][j] = mat[i][j - 1] 
                  
            # Else value is obtained considering two cases. 
            # a) All substrings without last character in S 
            # b) All substrings without last characters in 
            # both. 
            else: 
                mat[i][j] = (mat[i][j - 1] + 
                             mat[i - 1][j - 1]) 
  
    return mat[m][n]

#Function to count normal k mers
def count_(string, substring): 
    # Initialize count and start to 0 
    count = 0
    start = 0
  
    # Search through the string till 
    # we reach the end of it 
    while start < len(string): 
  
        # Check if a substring is present from 
        # 'start' position till the end 
        flag = string.find(substring, start) 
  
        if flag != -1: 
            # If a substring is present, move 'start' to 
            # the next position from start of the substring 
            start = flag + 1
  
            # Increment the count 
            count += 1
        else: 
            # If no further substring is present 
            # return the value of count 
            return count 
         
# starts training in CNN model
def run_model(lower_bound, upper_bound, max_len, dataset, **kwargs):
    '''load data into the playground'''
    X, y = preprocess_data(lower_bound, upper_bound,max_len, dataset, max_len)
    OUTPATH = os.path.join(basedir,
                               'Results/RNATracker-10foldcv/' + args.dataset+ '/' + str(datetime.datetime.now()).split('.')[0].replace(':', '-').replace(' ', '-') + '-' + args.message + '/')
    if not os.path.exists(OUTPATH):
            os.makedirs(OUTPATH)
    print("Build RNALocator")
    #Load protein information
    if(dataset == "cefra-seq"):
        print("Cefra-seq PPI inforamtion")
        with open('ppiMatrixScoress.npy', 'rb') as f:
            ppi = np.load(f)
        nb_classes = 4
        
    if(dataset == "rnalocate"):
        print("Loadin ppi info")
        with open('ppiHomosapRNALocate.npy', 'rb') as f:
            ppi = np.load(f)
        nb_classes = 5
        
        
        
    # Do feature importance with a regressor baseline
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LinearRegression
    
    from matplotlib import pyplot
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    scalerPPI = preprocessing.MinMaxScaler()
    pca = PCA(n_components=500)
    ppi = scaler.fit_transform(ppi)
    ppiPCA = pca.fit_transform(ppi)
    print("Done PCA")
    newData = ppiPCA

    #Compute novel distance based k mers

    newmer = []
    for k in range(0,9):
        i = "x" * k
        newAA = "A"+i+ "A"
        newAC = "A"+i+ "C"
        newAG = "A"+i+ "G"
        newAT = "A"+i+ "T"
        newmer.append(newAA)
        newmer.append(newAC)
        newmer.append(newAG)
        newmer.append(newAT)
        newCA = "C"+i+ "A"
        newCC = "C"+i+ "C"
        newCG = "C"+i+ "G"
        newCT = "C"+i+ "T"
        newmer.append(newCA)
        newmer.append(newCC)
        newmer.append(newCG)
        newmer.append(newCT)
        newGA = "G"+i+ "A"
        newGC = "G"+i+ "C"
        newGG = "G"+i+ "G"
        newGT = "G"+i+ "T"
        newmer.append(newGA)
        newmer.append(newGC)
        newmer.append(newGG)
        newmer.append(newGT)
        newTA = "T"+i+ "A"
        newTC = "T"+i+ "C"
        newTG = "T"+i+ "G"
        newTT = "T"+i+ "T"
        newmer.append(newTA)
        newmer.append(newTC)
        newmer.append(newTG)
        newmer.append(newTT)

    
    dictmers = { newmer[i] : i for i in range(0, len(newmer) ) }
   
    kmerscounted = []
    kdist = np.zeros(len(dictmers))
    print(len(dictmers))
    
    import regex as re
    
    for i in X:
        i = str(i)
        for j in range(0,2):
            k= str(j) 
            pattern = "A[A,C,G,T]{" +k+ "}A"
            matches = re.findall(pattern,i,overlapped=True)
            key = "A" + "x"*(j) + "A"
            kdist[dictmers[key]] = len(matches) / len(i)
        for j in range(0,9):
            k= str(j)
            pattern = "A[A,C,G,T]{" +k+ "}C"
            matches = re.findall(pattern,i,overlapped=True)
            key = "A" + "x"*(j) + "C"
            kdist[dictmers[key]] = len(matches) / len(i)
        for j in range(0,9):
            k= str(j)
            pattern = "A[A,C,G,T]{" +k+ "}G"
            key = "A" + "x"*(j) + "G"
            kdist[dictmers[key]] = len(matches) / len(i)
        for j in range(0,9):
            k= str(j)
            pattern = "A[A,C,G,T]{" +k+ "}T"
            matches = re.findall(pattern,i,overlapped=True)
            key = "A" + "x"*(j) + "T"
            kdist[dictmers[key]] = len(matches) / len(i)
        for j in range(0,9):
            k= str(j)
            pattern = "C[A,C,G,T]{" +k+ "}A"
            matches = re.findall(pattern,i,overlapped=True)
            key = "C" + "x"*(j) + "A"
            kdist[dictmers[key]] = len(matches) / len(i)
        for j in range(0,9):
            k= str(j)
            pattern = "C[A,C,G,T]{"+k+"}G"
            matches = re.findall(pattern,i,overlapped=True)
            key = "C" + "x"*(j) + "G"
            kdist[dictmers[key]] = len(matches) / len(i)
        for j in range(0,9):  
            k= str(j)
            pattern = "C[A,C,G,T]{"+k+"}C"
            matches = re.findall(pattern,i,overlapped=True)
            key = "C" + "x"*(j) + "C"
            kdist[dictmers[key]] = len(matches) / len(i)
        for j in range(0,9):
            k= str(j)
            pattern = "C[A,C,G,T]{"+k+"}T"
            matches = re.findall(pattern,i,overlapped=True)
            key = "C" + "x"*(j) + "T"
            kdist[dictmers[key]] = len(matches) / len(i)
        for j in range(0,9):
            k= str(j)
            pattern = "G[A,C,G,T]{"+k+"}A"
            matches = re.findall(pattern,i,overlapped=True)
            key = "G" + "x"*(j) + "A"
            kdist[dictmers[key]] = len(matches) / len(i)
        for j in range(0,9):
            k= str(j)
            pattern = "G[A,C,G,T]{"+k+"}C"
            matches = re.findall(pattern,i,overlapped=True)
            key = "G" + "x"*(j) + "C"
            kdist[dictmers[key]] = len(matches) / len(i)
        for j in range(0,9):
            k= str(j)
            pattern = "G[A,C,G,T]{"+k+"}G"
            matches = re.findall(pattern,i,overlapped=True)
            key = "G" + "x"*(j) + "G"
            kdist[dictmers[key]] = len(matches) / len(i)
        for j in range(0,9):
            k= str(j)
            pattern = "G[A,C,G,T]{"+k+"}T"
            matches = re.findall(pattern,i,overlapped=True)
            key = "G" + "x"*(j) + "T"
            kdist[dictmers[key]] = len(matches) / len(i)
        for j in range(0,9):
            k= str(j)
            pattern = "T[A,C,G,T]{"+k+"}A"
            matches = re.findall(pattern,i,overlapped=True)
            key = "T" + "x"*(j) + "A"
            kdist[dictmers[key]] = len(matches) / len(i)
        for j in range(0,9):
            k= str(j)
            pattern = "T[A,C,G,T]{"+k+"}C"
            matches = re.findall(pattern,i,overlapped=True)
            key = "T" + "x"*(j) + "C"
            kdist[dictmers[key]] = len(matches) / len(i)
        for j in range(0,9):
            k= str(j)
            pattern = "T[A,C,G,T]{"+k+"}G"
            matches = re.findall(pattern,i,overlapped=True)
            key = "T" + "x"*(j) + "G"
            kdist[dictmers[key]] = len(matches) / len(i)
        for j in range(0,9):
            k= str(j)
            pattern = "T[A,C,G,T]{"+k+"}T"
            matches = re.findall(pattern,i,overlapped=True)
            key = "T" + "x"*(j) + "T"
            kdist[dictmers[key]] = len(matches) / len(i)
                
        
            
        kmerscounted.append(kdist)
        kdist = np.zeros(len(dictmers))
       


    
    Xlen=[]
    
    kmers =np.array(kmerscounted)
 
    print("shape of k mers data: ",np.shape(kmers)) 
    
    # Add normal k mer as well
    
    with open('5mers.txt') as f:
        five = [line.rstrip() for line in f]

    
    newmer = five 
    dictmers = { newmer[i] : i for i in range(0, len(newmer) ) }
    kmerscounted = []
    kdist = np.zeros(len(dictmers))
    for i in X:
        for j in dictmers:
            i = str(i)
            pattern= str(j) 
            matches = re.findall(pattern,i,overlapped=True)
            kdist[dictmers[pattern]] = len(matches) / len(i)
        kmerscounted.append(kdist)
        kdist = np.zeros(len(dictmers))
    
    kmers_normal = np.array(kmerscounted)
    
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    scalerPPI = preprocessing.MinMaxScaler()
    pca = PCA(n_components=500)
    ppi = scaler.fit_transform(ppi)
    ppiPCA = pca.fit_transform(ppi)
    #print("Explained variance by PCA for PPI is", np.sum(pca.explained_variance_ratio_))
    #kmersData = np.concatenate((kmers,kmers_normal), axis = 1)
    #newData = np.concatenate((kmers, kmers_normal,ppiPCA), axis = 1)
    newData = ppiPCA
    newData = scaler.fit_transform(newData)
    #Model tobe applied on the whole data and further be tesetd on independed dataset
    print("Running model on the whole dataset")
    OUTPATH = os.path.join(basedir,
                               'Results/RNATracker-10foldcv/')
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)
    
    import random
    import pandas
    random.seed(1234)
    allframes = []
    for j in range(1):
        OUTPATH = os.path.join(basedir,
                               'Results/RNATracker-10foldcv/' + args.dataset+ '/' + str(datetime.datetime.now()).split('.')[0].replace(':', '-').replace(' ', '-') + '-' + args.message + '/')
        if not os.path.exists(OUTPATH):
            os.makedirs(OUTPATH)
        myindex = j
        state = random.randint(0,3000)
        #print("seed is", seed)
        kf = KFold(n_splits=10, shuffle=True, random_state=state)
        folds = kf.split(newData, y)
        for i, (train_indices, test_indices) in enumerate(folds):
            newData[train_indices] = scaler.fit_transform(newData[train_indices])
            newData[test_indices] = scaler.fit_transform(newData[test_indices])
            print('Evaluating KFolds {}/10'.format(i + 1))
            # from Models.RBPBindingModel import RBPBinder
            # model = RBPBinder(max_len, nb_classes, OUTPATH)
            from neural_network_predictor import RNALocator
            model = RNALocator(max_len, nb_classes, OUTPATH, kfold_index=i)# initialize
            print("Build RNALocator")
            model.build_neural_network(np.shape(newData)[1])
            model.train(newData[train_indices], y[train_indices], batch_size, kwargs['epochs'])
            model.evaluate(newData[test_indices],y[test_indices], dataset)
        mydataframe = evaluate_folds.evaluate_folds(OUTPATH, dataset,myindex)
        allframes.append(mydataframe)
    
    import pickle
    with open("savedfolds.txt", "wb") as fp:
        pickle.dump(allframes, fp)
        
    result = pd.concat(allframes)
    with open("savedfoldsConcat.txt", "wb") as fp:
        pickle.dump(result, fp)
    
    print(result)    
           
'''Always draw scatter plots for each experiment we run'''
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''Model parameters'''
    parser.add_argument('--lower_bound', type=int, default=0, help='set lower bound on sample sequence length')
    parser.add_argument('--upper_bound', type=int, default= 40000, help='set upper bound on sample sequence length') #default=4000
    parser.add_argument('--max_len', type=int, default=40000,
                        help="dummy, pad or slice sequences to a fixed length in preprocessing")
    #parser.add_argument('--nb_classes', type=int, default=5, help='number of locations in each dataset')
    parser.add_argument('--dataset', type=str, default='cefra-seq', choices=['cefra-seq', 'rnalocate'],
                        help='choose from cefra-seq and rnalocate')
    parser.add_argument('--epochs', type=int, default=300, help='')

    parser.add_argument("--message", type=str, default="", help="append to the dir name")
    parser.add_argument("--load_pretrain", action="store_true",
                        help="load pretrained CNN weights to the first convolutional layers")
    parser.add_argument("--weights_dir", type=str, default="",
                        help="Must specificy pretrained weights dir, if load_pretrain is set to true. Only enter the relative path respective to the root of this project.")
    parser.add_argument("--randomization", type=int, default=None,
                        help="Running randomization test with three settings - {1,2,3}.")
    # parser.add_argument("--nb_epochs", type=int, default=20, help='choose the maximum number of iterations over training samples')
    args = parser.parse_args()

        
    OUTPATH = os.path.join(basedir,
                               'Results/RNALocator-10foldcv/' + args.dataset+ '/' + str(datetime.datetime.now()).split('.')[0].replace(':', '-').replace(' ', '-') + '-' + args.message + '/')
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)
    print('OUTPATH:', OUTPATH)
    #del args.message

    args.weights_dir = os.path.join(basedir, args.weights_dir)

    for k, v in vars(args).items():
        print(k, ':', v)

    
    run_model(**vars(args))
