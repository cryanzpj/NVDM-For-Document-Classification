import numpy as np
from collections import Counter,defaultdict


def vocab_to_dict(file):
    vocab_list = open(file).readlines()
    return dict(np.array(map(lambda x: tuple(x.strip().split(' ')[:2]),vocab_list),dtype='|S51,i4'))

def make_corp(file,vocab):
    file = open(file)
    corpus = []
    c_id,c = 0,Counter()

    while True:
        next_ = file.readline()
        if next_ == '':
            break
        if next_ == '\n':
            continue
        else:
            next_ = next_.strip()
            if next_.split(' ')[0] != '.I'  and next_.split(' ')[0] != '.W' :
                c += Counter(next_.split(' '))
            elif next_.split(' ')[0] == '.I':
                if c_id == 0:
                    c_id,c = int(next_.split(' ')[1]),Counter()
                else:
                    for key in c.keys():
                        c[vocab[key]] = c.pop(key)
                    corpus.append(map(lambda x: (x[0],x[1]),c.items()))
                    c_id,c = int(next_.split(' ')[1]),Counter()
    for key in c.keys():
        c[vocab[key]] = c.pop(key)
    corpus.append(map(lambda x: (x[0],x[1]),c.items()))

    return corpus


def make_label(file_):
    data = np.loadtxt(file_,dtype = object)
    label_dict = dict(zip(set(data[:,0]),range(len(set(data[:,0])))))
    data_dict = defaultdict(list)
    for i in data:
        data_dict[int(i[1])].append(label_dict[i[0]])

    labels = np.zeros((len(set(data[:,1])),len(set(data[:,0]))),dtype = int)

    for i,j in enumerate(data_dict.items()):
        labels[i,j[-1]] = 1

    return labels


