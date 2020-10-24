import numpy as np
import os
import scipy.stats as stats
import argparse
import sys
import scipy


basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#from Scripts.RNALocator import preprocess_data
'''plt.style.use('ggplot')
matplotlib.rcParams.update({'font.family': 'Times New Roman', 'font.size': 36, 'font.weight': 'light', 'figure.dpi': 350})'''

def label_dist(dist):
    '''
    dummy function
    :param dist:
    :return:
    '''
    assert (len(dist) == 4)
    return np.array(dist) / np.sum(dist)

def evaluate_folds(expr_path, dataset, myindex):
    print("OS path is",os.path)
    if not os.path.isabs(expr_path):
        expr_path = os.path.join(basedir, expr_path)
    print("expr_path", expr_path)
    

    print('Loading experiments at', expr_path)
    '''load kfolds data'''
    
    if not os.path.exists(os.path.join(expr_path, 'scatter')):
        os.makedirs(os.path.join(expr_path, 'scatter'))
        

    '''load predictions made for different folds'''
    y_test = []
    y_pred = []
    for kfold_index in range(10):
        if os.path.exists(os.path.join(expr_path, 'y_label_fold_{}.npy'.format(kfold_index))):
            y_test.append(np.load(os.path.join(expr_path, 'y_label_fold_{}.npy'.format(kfold_index))))
            y_pred.append(np.load(os.path.join(expr_path, 'y_predict_fold_{}.npy'.format(kfold_index))))
        else:
            break
    y_test = np.concatenate(y_test)
    y_pred = np.concatenate(y_pred)



 
    #For mRNALoc results
    '''import pickle
    f = open("realTargetsHuman.pkl","rb")
    y_test = pickle.load(f)

    f = open("mRNALocMethod_results.txt", "r")
    y_pred_temp = [line.strip("\n") for line in f.readlines()]
    f.close()
    y_pred_temp.pop(0)
    y_pred = []
    for item in y_pred_temp:
        y_pred.append(item.split("\t")[2])


    y_pred = np.asarray(y_pred)
    del_ = []
    for indx, item in enumerate(y_pred):
        if(item == 'No Location Found'):
            #print(item)
            del_.append(indx)
      
    y_pred = np.delete(y_pred, del_, axis=0)   
    y_test =   np.delete(y_test, del_, axis=0)
    from keras.utils import to_categorical
    total_locs = ["Cytoplasm", "Endoplasmic_Reticulum", "Extracellular_region", "Mitochondria", "Nucleus"]
    ### map each color to an integer
    mapping = {}
    for x in range(len(total_locs)):
      mapping[total_locs[x]] = x
    
    # integer representation
    for x in range(len(y_pred)):
      y_pred[x] = mapping[y_pred[x]]
    
    one_hot_encode = to_categorical(y_pred)
    y_pred = one_hot_encode'''
    
  #For RNALocate dataset
    y_test = np.argmax(y_test, axis = 1)

    loc0 = 0
    loc1 = 0
    loc2 = 0
    loc3 = 0
    loc4 = 0
    print("Count in y_test")
    for i in y_test:
        if( i == 0):
            loc0 = loc0 +1
        if( i == 1):
            loc1 = loc1 +1
        if( i == 2):
            loc2 = loc2 +1
        if( i == 3):
            loc3 = loc3 +1
        if( i == 4):
            loc4 = loc4 +1
     
    print(loc0,loc1,loc2,loc3,loc4)
    y_pred_list = []
    idx = 0
    count_strong = 0
    loc_count = 0
    threshold_vals = [0.01, 0.015, 0.02, 0.05, 0.1, 0.15, 0.2]
    threshold_vals = np.asanyarray(threshold_vals)

    acc = []
    th = 0.15
    idx = 0
    for idx , item in enumerate(y_pred):
        real = y_test[idx]
        y_pred = np.argmax(item)

        y_pred_list.append(y_pred)
    
    #for th in threshold_vals:
   


    print("More than 3 locs", loc_count)  
    y_pred = y_pred_list

    '''
    loc0 = 0
    loc1 = 0
    loc2 = 0
    loc3 = 0
    loc4 = 0
    print("Count in y_pred")
    for i in y_pred:
        if( i == 0):
            loc0 = loc0 +1
        if( i == 1):
            loc1 = loc1 +1
        if( i == 2):
            loc2 = loc2 +1
        if( i == 3):
            loc3 = loc3 +1
        if( i == 4):
            loc4 = loc4 +1
     
    print(loc0,loc1,loc2,loc3,loc4)'''

    locations = ["cytosol", "Endo_re", "Ex_re", "Mito", "Nucleus"]

    '''f = open("KFoldresult{}".format(myindex),"w")
    f.write()'''

  
    
    
    import pandas
    from sklearn.metrics import classification_report
    from sklearn import metrics
    
    report = classification_report(y_test, y_pred, digits = 10)
    print(report)
    return
    report = classification_report(y_test, y_pred, output_dict=True)
    mydataframe = pandas.DataFrame(report).transpose()
    from sklearn.metrics import matthews_corrcoef
    print("MCC",matthews_corrcoef(y_test, y_pred))
    print("Finished")
    from sklearn.metrics import confusion_matrix
    print( confusion_matrix(y_test, y_pred))
    
    '''fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=5)
    print("AUC",metrics.auc(fpr, tpr) )'''
        
    return mydataframe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--expr_path", type=str, default="",
                        help="Specify saved experiment folder. If path is relative, please make sure it's relative to the root folder.")
    parser.add_argument("--randomization", type=int, default=None,
                        help="Running randomization test with three settings - {1,2,3}.")
    parser.add_argument('--dataset', type=str, default='cefra-seq', choices=['cefra-seq', 'apex-rip'],
                        help='choose from cefra-seq and apex-rip')
    args = parser.parse_args()
    print(args.randomization)

    evaluate_folds(args.expr_path, args.dataset, 0)

