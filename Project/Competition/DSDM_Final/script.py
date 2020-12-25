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

# randoms = [0,11,21,32,41,51,61,71,81,91]

sim_options = {
    "name": "cosine",
    "user_based": True,  # Compute  similarities between items
}

target = 0.88380

while True:

    rand_int = np.random.randint(100000)
    trainset, valset = train_test_split(train, test_size=.25,random_state=rand_int)

    svd = SVD()
    print('Train                        ',end='\r')
    svd.fit(trainset)
    print('Predict                      ',end='\r')
    predictions = svd.test(valset)
    

    acc = accuracy.rmse(predictions,verbose=False)

    if abs(acc-target) < 0.014:
        print(rand_int,abs(acc-target))
    else:
        print(rand_int,abs(acc-target),end='\r')
        time.sleep(0.5)
