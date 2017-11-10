import pickle
import os

from datapoint import Datapoint


'''
    Dataset
      - ( structured_data )
      - ( list_of_dicts )

'''
class Dataset(object):

    def __init__(self, pipe=None, flush=False):

        if not flush and self.read():
            pass

        else:
            traindata, testdata = pipe()
            # create data points
            self.trainset = self.sort([ Datapoint(item) # sort list
                for item in traindata ])
            self.testset  = self.sort([ Datapoint(item) 
                    for item in testdata  ])

            # save to disk
            self.write()

    def sort(self, data):
        return sorted(data, 
                key = lambda x : x.context_len())

    def write(self):
        with open('.cache/train.pkl', 'wb') as f:
            pickle.dump(self.trainset, f)
        with open('.cache/test.pkl', 'wb') as f:
            pickle.dump(self.testset, f)

    def read(self):
        # check if cache dir exists
        if not os.path.exists('.cache'):
            os.makedirs('.cache')

        try:
            with open('.cache/train.pkl', 'rb') as f:
                self.trainset = pickle.load(f)
            with open('.cache/test.pkl', 'rb') as f:
                self.testset = pickle.load(f)
            return True
        except:
            pass
