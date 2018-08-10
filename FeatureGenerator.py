import os,sys
from SenFeatures import SentFeatures

class Feature:
    def __init__(self,id,val):
        self.id = id
        self.val = val

    def __str__(self):
        return str(self.id)+":"+str(self.val)

def word_matches(h, ref):
    return sum(1 for w in h if w in ref)

def precs(n,t):
    return float(n)/t

def rec(n,t):
    return float(n)/t

def f1(prec,rec):
    if prec+rec == 0:
        return 0
    else:
        return 2*prec*rec/(prec+rec)

class FeatureGenerator:
    def __init__(self,s1,s2):
        self.s1=SentFeatures(s1)
        self.ref=SentFeatures(s2)
        self.generate_features()

    def generate_features(self):
        self.feature_list=[]
        dict_keys = ["unigram","bigram","trigram","quadgram","unigram_p","bigram_p","trigram_p","quadgram_p"]
        id = 1
        for key in dict_keys:
            cand = self.s1.h_dict[key]
            ref = self.ref.h_dict[key]
            rset = set(ref)
            cand_match = word_matches(cand, rset)
            total_ref = len(ref)
            prec = precs(cand_match,len(cand))
            recall = rec(cand_match,total_ref)
            f1_measure = f1(prec,recall)
            self.feature_list.append(Feature(id,prec))
            id+=1
            self.feature_list.append(Feature(id,recall))
            id+=1
            self.feature_list.append(Feature(id,f1_measure))
            id+=1
        average_ngram_precision = (self.feature_list[1].val + self.feature_list[4].val + self.feature_list[7].val + self.feature_list[10].val)/4
        self.feature_list.append(Feature(id,average_ngram_precision))
        id+=1
        word_count = float(self.s1.wordcount)/self.ref.wordcount
        self.feature_list.append(Feature(id, word_count))
        id+=1
        function_words = float(self.s1.funcwords)/self.ref.wordcount
        self.feature_list.append(Feature(id, function_words))
        id+=1
        punc_count = float(self.s1.punc_count)/self.ref.wordcount
        self.feature_list.append(Feature(id, punc_count))
        id += 1
        content_words = float(self.s1.contentwords)/self.ref.wordcount
        self.feature_list.append(Feature(id, content_words))
        id += 1
        dict_keys_m = ["unigram_m", "bigram_m", "trigram_m", "quadgram_m"]
        for key in dict_keys_m:
            cand = self.s1.h_dict[key]
            ref = self.ref.h_dict[key]
            rset = set(ref)
            cand_match = word_matches(cand, rset)
            total_ref = len(ref)
            prec = precs(cand_match,len(cand))
            self.feature_list.append(Feature(id, prec))
            id+=1

    def __str__(self):
        formstring=""
        for feature in self.feature_list:
            formstring += " "+str(feature)
        return formstring

if __name__ == '__main__':
    pass
