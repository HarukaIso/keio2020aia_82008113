from . import *

class Sonar_Data:

    def __init__(self, data_file_path='', data_file_name='sonar_data.pkl'):

        self.data = []

        self._i = 0
        with open('/Users/HARU/lecture/data/sonar_data.pkl', mode='rb') as f:
            mm = pkl.load(f) # dataset is dictionary

        value = list(mm.values())
        #key = list(mm.keys())
        m = list(value[0]) # 111sample, vector of length 60
        #m = mvalue.flatten() #vector of x, label is mine
        r = list(value[1]) # 97sample, vector of length 60
        #r = rvalue.flatten() #vector of x, label is rock
        mdata = [(x, 1) for x in m]
        rdata = [(x, 0) for x in r]
        #xym = [[val, 1] for val in m]
        #xyr = [[val, 0] for val in r]
        #combine
        self.data = mdata + rdata #data = [[vec.x1, y1], ..., [vec.xn, yn]] n=111+97

        self.shuffle()

    def __iter__(self):

        return self

    def __next__(self):
        if self._i == len(self.data):
            raise StopIteration()

        x, y = self.data[self._i] #unpack
        self._i += 1
        return (x, y)

    def shuffle(self):

        random.shuffle(self.data)

    def __len__(self):

        return len(self.data)
