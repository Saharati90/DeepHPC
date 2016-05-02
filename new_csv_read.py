# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 01:37:25 2016
"""

def MyInput(csvpath, batch_size, indicator):
    '''This function gets the full path of a csv file, the desired batch size and 
    an indicator which shows the index of batch chunk we want to grab and
    returns the batch_data and its corresponding batch_label
    csvpath: the full path of a csv file
    batch_size: desired batch size
    indicator = an indicator for the desired batch chunk with initial value of 0'''
    import csv
    import random
    import numpy as np

    SEED = 448
    mylist = []
    batch_data = []
    batch_label = []
    #with open(csvpath, 'rb') as csvfile:
    with open(csvpath, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            mylist.append(row[0].split(','))
            #if '' in row[0].split(','):
            #   myarray
            #else:
            #   myarray.append(row[0].split(','))
    random.seed(SEED)
    random.shuffle(mylist)
    for i in range(batch_size):
        batch_data.append(mylist[i + batch_size * indicator][2:133])
        batch_label.append(mylist[i + batch_size * indicator][1])
        
    batch_data = np.array(batch_data).astype(np.float32)
    batch_label = np.array(batch_label).astype(np.float64)
    
    
    return batch_data, batch_label
    
    
# test:
#csvpath = '/home/pooya/Desktop/hpc/new_train.csv'
#batch_data, batch_label = MyInput(csvpath, 20, 0)
#print batch_data
#print batch_label
