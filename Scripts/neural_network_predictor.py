
from tensorflow.keras import regularizers
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.python.keras.layers.convolutional import Convolution1D, MaxPooling1D
from tensorflow.keras.layers import Embedding, BatchNormalization, LSTM, Bidirectional, Input, Concatenate, Multiply, Dot, Reshape, Activation, Lambda, Masking
from tensorflow.keras.models import Model
from six.moves import range
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import subprocess
from tensorflow.keras import backend as K
from matplotlib.ticker import MaxNLocator
from tensorflow.python.keras.initializers import random_normal
import os
import scipy.stats as stats
import csv
import sys
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from tensorflow.keras import optimizers
import tensorflow.keras
from tensorflow.keras.layers import concatenate
from sklearn.svm import SVR

'''def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )'''

'''def myAccuracy(y_true, y_pred):
    return r2_score(y_true, y_pred)'''

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

plt.style.use('ggplot')
matplotlib.rcParams.update(
    {'font.family': 'Times New Roman', 'font.size': 36, 'font.weight': 'light', 'figure.dpi': 350})

OUTPATH = None
pretrained_model_score = [0.8, 0.62, 0.88, 0.89, 0.63, 0.94, 0.97, 0.94, 0.62, 0.89, 0.92,
                          0.92, 0.95, 0.96, 0.72, 0.96, 0.98, 0.77, 0.78, 0.7, 0.83, 0.87,
                          0.96, 0.97, 0.9, 0.97, 0.93, 0.94, 0.9, 0.96, 0.95]


class FigureCallback(Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []

        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6))
        plt.pause(0.001)
        fig.subplots_adjust(hspace=0.4)
        plt.pause(0.001)
        ax1.set_title('Loss')
        plt.pause(0.001)
        ax1.set_xlabel('Epoch')
        plt.pause(0.001)
        ax1.set_ylabel('Loss')
        plt.pause(0.001)
        ax2.set_title('Accuracy')
        plt.pause(0.001)
        ax2.set_xlabel('Epoch')
        plt.pause(0.001)
        ax2.set_ylabel('Acc')
        plt.pause(0.001)

        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2

        self.epochs = 0

    def on_epoch_end(self, epoch, logs={}):
        self.epochs += 1
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        self.ax1.clear()
        self.ax2.clear()
        self.ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Acc')
        self.ax1.plot(np.arange(1, self.epochs + 1, 1, dtype=int), self.loss, label='Training Loss')
        self.ax1.plot(np.arange(1, self.epochs + 1, 1, dtype=int), self.val_loss, label='Validation Loss')
        self.ax2.plot(np.arange(1, self.epochs + 1, 1, dtype=int), self.acc, label='Training Accuracy')
        self.ax2.plot(np.arange(1, self.epochs + 1, 1, dtype=int), self.val_acc, label='Validation Accuracy')
        self.ax1.legend()
        self.ax2.legend()

        plt.draw()
        plt.pause(0.1)

    def on_train_end(self, logs={}):
        # save graph
        plt.savefig(OUTPATH + 'Train_Val.png')

class RNALocator:

    def __init__(self, max_len, nb_classes, save_path, kfold_index):
        print("Constructing RNALocator class")
        print("Number of classes is", nb_classes)
        self.max_len = max_len
        self.nb_classes = nb_classes
        self.is_built = False
        global OUTPATH
        OUTPATH = save_path
        self.kfold_index = kfold_index
        

   
                   
    def build_autoencoder(self, kmer_vector):
        
        print("Building auto encoder model with 3 layers")
      
        self.is_built = True
        self.bn = False
        
        
        input_dim = kmer_vector
        encoding_dim = 200
        compression_factor = float(input_dim) / encoding_dim
        print("Compression factor: %s" % compression_factor)

        input_data = Input(shape=(input_dim,))
        encoded = Dense(1000, activation = 'relu')(input_data)
        encoded = Dense(500, activation = 'relu')(encoded)
        encoded = Dense(encoding_dim, activation = 'relu')(encoded)
        decoded = Dense(500, activation = 'relu')(encoded)
        decoded = Dense(1000, activation = 'relu')(encoded)
        decoded = Dense(input_dim, activation = 'linear')(decoded)
         
         

        nadam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # Decoder Layers
        autoencoder = Model(input_data, decoded)
        self.model = autoencoder
        self.model.compile(
                loss= 'mse',
                optimizer=nadam,
                metrics=['acc']
                )
        
    
        
        
        '''self.model.compile(
            loss='kld',
            optimizer='nadam',
            metrics=['acc']
        )'''
        self.is_built = True
        self.bn = False
        self.model.summary()
        
    def build_SVR(self,):
        regressor = SVR(kernel = 'rbf')
        self.model = regressor
        self.is_built = True
        self.bn = False
        self.model.summary()

    def build_neural_network(self,input_dim):
        input_data = Input(shape=(input_dim,))
        #input_data = Dropout(0.1)(input_data)
        first = Dense(200,activation = 'relu')(input_data)
        first_out = Dropout(0.2)(first)
        '''second = Dense(180,activation = 'relu')(first_out)
        second_out = Dropout(0.2)(second)
        third = Dense(150,activation = 'relu')(second_out)
        third_out = Dropout(0.2)(third)'''
        '''forth = Dense(50,activation = 'relu')(third_out)
        forth_out = Dropout(0.1)(forth)'''
        output = Dense(4,activation = 'softmax')(first_out)
        model_ = Model(input_data, output)
        loss_ = tf.keras.losses.CategoricalCrossentropy()
        self.model = model_
        self.model.compile(
                loss= 'kld',
                #optimizer='nadam',
                optimizer='nadam',
                metrics=['acc']
                )

        self.is_built = True
        self.bn = False
        #self.model.load_weights('weights_fold_9.h5')
        self.model.summary()
        
       

    
    @classmethod
    def acc(cls, y_true, y_pred):
        '''
        soft-accuracy; should never be used.
        :param y_true: target probability mass of mRNA samples
        :param y_pred: predcited probability mass of mRNA samples
        :return: averaged accuracy
        '''
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

    def get_feature(self, X):
        '''
        K.learning_phase() returns a binary flag
        The learning phase flag is a bool tensor (0 = test, 1 = train)
        to be passed as input to any Keras function that
        uses a different behavior at train time and test time.
        '''
        inputs = [K.learning_phase()] + [self.model.inputs[0]]
        _convout1_f = K.function(inputs, [self.model.layers[2].output])  # output of first convolutional filter
        activations = _convout1_f([0] + [X])

        return activations

    

    def get_masking(self, X):
        if self.bn:
            layer = 14
        else:
            layer = 12
        inputs = [K.learning_phase()] + [self.model.inputs[0]]
        _attention_f = K.function(inputs, [self.model.layers[layer].output])

        return _attention_f([0] + [X])

    
       
    def train(self, x_train, y_train, batch_size, epochs=300):
        if not self.is_built:
            print('Run build_model() before calling train opertaion.')
            return
        
        
        size_train = len(x_train)
        x_valid = x_train[int(0.9 * size_train):]
        y_valid = y_train[int(0.9 * size_train):]
        x_train = x_train[:int(0.9 * size_train)]
        y_train = y_train[:int(0.9 * size_train)]
        
        # early_stopping = EarlyStopping(monitor='val_loss', patience=7)
        best_model_path = OUTPATH + 'weights_fold_{}.h5'.format(self.kfold_index)
        model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, verbose=1)
        #evaluating in Keras
        print(self.model.evaluate(x_train, y_train, batch_size=batch_size))
        print(self.model.evaluate(x_valid, y_valid, batch_size=batch_size))
        hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                              validation_data=(x_valid, y_valid), callbacks=[model_checkpoint], shuffle=True)
        # load best performing model
        self.model.load_weights(best_model_path)
        Train_Result_Optimizer = hist.history
        Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
        Train_Loss = np.array([Train_Loss]).T
        Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
        Valid_Loss = np.asarray([Valid_Loss]).T
        Train_Acc = np.asarray(Train_Result_Optimizer.get('acc'))
        Train_Acc = np.array([Train_Acc]).T
        Valid_Acc = np.asarray(Train_Result_Optimizer.get('val_acc'))
        Valid_Acc = np.asarray([Valid_Acc]).T
        np.savetxt(OUTPATH + 'Train_Loss_fold_{}.txt'.format(self.kfold_index), Train_Loss, delimiter=',')
        np.savetxt(OUTPATH + 'Valid_Loss_fold_{}.txt'.format(self.kfold_index), Valid_Loss, delimiter=',')
        np.savetxt(OUTPATH + 'Train_Acc_fold_{}.txt'.format(self.kfold_index), Train_Acc, delimiter=',')
        np.savetxt(OUTPATH + 'Valid_Acc_fold_{}.txt'.format(self.kfold_index), Valid_Acc, delimiter=',')
        return hist
        #PRINT LSTM OUTPUT
        #print("output is ", self.model.layer[2].output)
    def evaluate(self, x_test, y_test, dataset):
        
        score, acc = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score)
        print('Test accuracy:', acc)
        
        y_pred = self.model.predict(x_test)
        # save label and predicted values for future plotting
        np.save(OUTPATH + 'ind_labelNew.npy'.format(self.kfold_index), y_test)
        np.save(OUTPATH + 'ind_predNew.npy'.format(self.kfold_index), y_pred)
        
        import seaborn as sns
        import scipy
        spearman = np.zeros(len(y_test))
        k_spearman = 0
    
    
        for k_spearman in range(len(y_test)):
            rank , pvalue = scipy.stats.spearmanr(y_test[k_spearman], y_pred[k_spearman])
            spearman[k_spearman] = rank
            
        print("shape of spearman", np.shape(spearman))
        count_spearman = np.zeros(4)
        count_spearman2 = np.zeros(4)
        spearmanexact = 0
        i = 0
        for i in range (len(spearman)):
           
            if spearman[i] > 0.7:
                count_spearman[0] = count_spearman[0] + 1
            if spearman[i] >=0.5:
                count_spearman2[0] = count_spearman2[0] + 1
            if spearman[i] > 0.8:
                count_spearman[1] = count_spearman[1] + 1
            if spearman[i] >= 0.6:
                count_spearman2[1] = count_spearman2[1] + 1
            if spearman[i] > 0.9:
                count_spearman[2] = count_spearman[2] + 1
            if spearman[i] >= 0.7:
                count_spearman2[2] = count_spearman2[2] + 1
            if spearman[i] > 0.95:
                count_spearman[3] = count_spearman[3] + 1
            if spearman[i] >= 0.8:
                count_spearman2[3] = count_spearman2[3] + 1
            if (spearman[i] == 1):
                spearmanexact = spearmanexact + 1
          
        print("more than 0.7, 0.8, 0.9, 0.95", count_spearman)
        print("more equal than 0.5, 0.6, 0.7, 0.8", count_spearman2)
        print("spearman = 1", spearmanexact)
    
        if dataset == 'apex-rip':
            locations = ['KDEL', 'Mito', 'NES', 'NLS']
        elif dataset == 'cefra-seq':
            locations = ["cytosol", "insoluble", "membrane", "nucleus"]
        else:
            raise RuntimeError('No such dataset.')
        figures = []
        for i, loc in enumerate(locations):
            corr, pval = stats.pearsonr(y_pred[:, i], y_test[:, i])
            #corr, pval = stats.pearsonr(y_test[:, i],y_random[:, i])
             
            if(i == 0):
                print("Pearson correlation in Cytosol is ", corr)
                print("P-value  in  Cytosol is ", pval)
            if(i == 1):
                print("Pearson correlation in Insoluble is ", corr)
                print("P-value  in  Insoluble is ", pval)
            if( i == 2):
                print("Pearson correlation in Membrane is ", corr)
                print("P-value  in  Membrane is ", pval)
            if(i == 3):
                print("Pearson correlation in Nucleas is ", corr)
                print("P-value  in  Nucleas is ", pval)
            
                
            
            '''import pandas
            from sklearn.metrics import classification_report
            from sklearn import metrics
            y_test = np.argmax(y_test, axis = 1)
            idx = 0
            y_pred_list = []
            for idx , item in enumerate(y_pred):
                real = y_test[idx]
                y_pred = np.argmax(item)
    
                y_pred_list.append(y_pred)
          
            y_pred = y_pred_list
            report = classification_report(y_test, y_pred)
            print(report)
            report = classification_report(y_test, y_pred, output_dict=True)
            mydataframe = pandas.DataFrame(report).transpose()
            from sklearn.metrics import matthews_corrcoef
            print("MCC",matthews_corrcoef(y_test, y_pred))
            print("Finished")
            from sklearn.metrics import confusion_matrix
            print( confusion_matrix(y_test, y_pred))
            if dataset == 'rnalocate':
                locations = ['KDEL', 'Mito', 'NES', 'NLS']
            elif dataset == 'cefra-seq':
                locations = ["cytoplasm", "insoluble", "membrane", "nucleus"]
            else:
                raise RuntimeError('No such dataset.')
            #self.multiclass_roc_and_pr(y_test, y_predict, locations)'''
        return score, acc

    
    def saveModel(self):
        self.model.save('NeginMethod.hdf5')
        
    
