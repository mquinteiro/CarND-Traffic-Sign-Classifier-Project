# Load pickled data
import pickle
from sklearn.preprocessing import scale


# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)


X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


from skimage import exposure
import threading


class normalizeTh(threading.Thread):
    def __init__(self, image, idx):
        self.image = image
        self.output = image
        self.idx = idx
        threading.Thread.__init__(self)

    def run(self):
        # print("test")
        self.output = exposure.equalize_adapthist(self.image, clip_limit=0.03)


# I have 12 threads, use 10 to have two for other process we can also take 12 and reduce nice value.

import numpy as np


from tqdm import tqdm

def normalizeImgTh(X):
    numThreads = 10
    idx = 0
    dest = np.empty(X.shape)
    ths = []
    print(str(threading.activeCount()) + " Alive")
    for img in tqdm(X):
        # if we have all threads used, wait until fist is free
        if len(ths) >= numThreads:
            ths[0].join()
            dest[ths[0].idx] = ths[0].output
            del ths[0]
            #print("finishing normally")

        nTh = normalizeTh(img, idx)
        nTh.start()
        ths.append(nTh)
        idx += 1
        #delete all finished threads... garbage out
        for i in range(len(ths),0,-1):
            if not ths[i-1].is_alive():
                #ths[0].join()
                dest[ths[i-1].idx] = ths[i-1].output
                del ths[i-1]
                #print("Garbaje colector")
        #print(str(threading.activeCount()) + " Alive")

    # wait for all pending threads.
    while len(ths) > 0:
        ths[0].join()
        dest[ths[0].idx] = ths[0].output
    return dest


dest=normalizeImgTh(X_train)