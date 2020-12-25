import numpy as np 
import pandas as pd 

from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy
from surprise import KNNWithMeans
import time

train_df = pd.read_csv("training.txt", header=None)
reader = Reader(rating_scale=(1, 5))

train = Dataset.load_from_df(train_df[[0,1,2]], reader=reader)

sim_options = {
    "name": "cosine",
    "user_based": True,  # Compute  similarities between items
}

target = 0.89

file_opened = open('result.txt', 'r') 
count = 0

while True: 
    count += 1
  
    # Get next line from file 
    line = file_opened.readline() 
  
    # if line is empty 
    # end of file is reached 
    if not line: 
        break
    # print(line.split()[0]) 

    rand_int = int(line.split()[0])

    trainset, valset = train_test_split(train, test_size=.25,random_state=rand_int)

    knnwm = KNNWithMeans(sim_options=sim_options,verbose=False)
    print('Train                        ',end='\r')
    knnwm.fit(trainset)
    print('Predict                      ',end='\r')
    predictions = knnwm.test(valset,verbose=False)
    acc = accuracy.rmse(predictions,verbose=False)

    if abs(acc-target) < 0.014:
        print('CHECK',rand_int,abs(acc-target))
    else:
        print(rand_int,abs(acc-target))
        time.sleep(0.5)