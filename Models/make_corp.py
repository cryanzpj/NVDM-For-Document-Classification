import numpy as np
import scipy as sp
import re
from collections import Counter,defaultdict


def vocab_to_dict(file):
    vocab_list = open(file).readlines()
    return dict(np.array(map(lambda x: tuple(x.strip().split(' ')[:2]),vocab_list),dtype='|S51,i4'))

def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()




def make_corp(file,model,vocab = None):
    file = open(file)
    corpus = []
    c_id,c = 0,Counter()

    # making corpus for LDA model
    if vocab and model == 'LDA':
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
                            try:
                                vocab[key]
                            except KeyError:
                                c.pop(key)
                            else:
                                c[vocab[key]] = c.pop(key)
                        corpus.append(map(lambda x: (x[0],x[1]),c.items()))
                        c_id,c = int(next_.split(' ')[1]),Counter()
        for key in c.keys():
            try:
                vocab[key]
            except KeyError:
                c.pop(key)
            else:
                c[vocab[key]] = c.pop(key)
        corpus.append(map(lambda x: (x[0],x[1]),c.items()))

        return corpus
    #making data for NVDM model
    elif model == "NVDM":
        corpus_Count = []
        while True:
            next_ = file.readline()
            if next_ == '':
                break
            if next_ == '\n':
                continue
            else:
                next_ = next_.strip()
                if next_.split(' ')[0] != '.I'  and next_.split(' ')[0] != '.W' :
                    #vocab+= Counter(next_.split(' '))
                    corpus += next_.split(" ")
                    c+=Counter(next_.split(" "))
                elif next_.split(' ')[0] == '.I':
                    if c_id == 0:
                        c_id,c = int(next_.split(' ')[1]),Counter()
                    else:
                        corpus_Count.append(c)
                        c_id,c = int(next_.split(' ')[1]),Counter()
        corpus_Count.append(c)

        if not vocab:
            vocab = Counter(corpus)
            for i in vocab.keys():
                if clean_str(i) == "":
                    vocab.pop(i)
            vocab = dict(zip(np.asarray([("<UNK>",1)] + vocab.most_common(9999))[:,0],range(10000)))
        else:
            pass

        out = np.zeros((len(corpus_Count),10000))
        for i,j in enumerate(corpus_Count):
            for ele in j.keys():
                try: vocab[ele]
                except KeyError:
                    j['<UNK>'] += j.pop(ele)

            temp = np.array([[vocab[ele[0]], int(ele[1])]  for ele in j.items() ])
            temp = temp[np.argsort(temp[:,0])]
            out[i,temp[:,0]] = temp[:,1]

        if not vocab:
            return sp.sparse.csr_matrix(out),vocab
        else:
            return sp.sparse.csr_matrix(out)



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

