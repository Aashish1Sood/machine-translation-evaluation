from string import punctuation
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

stop_wordlist = stopwords.words('english')

class SentFeatures:
    def __init__(self,sent):
        self.sent = sent
        self.t_sent = self.preprocess()
        self.h_dict = {}
        self.generate_sent_features()

    def preprocess(self):
        self.punc_count=0
        t_list = self.sent.split(" ")
        new_list = []
        self.t_sent_upper = []
        for word in t_list:
            if word[-1] in punctuation:
                self.punc_count+=1
                word = word[:-1]
                if len(word) > 0:
                    new_list.append(word.lower())
                    self.t_sent_upper.append(word)
            else:
                new_list.append(word.lower())
                self.t_sent_upper.append(word)
        return new_list

    def generate_sent_features(self):
        self.wordcount = len(self.t_sent)
        self.pos = pos_tag(self.t_sent)
        self.pos_list = [item[1] for item in self.pos]
        if self.wordcount<4:
            self.ngram_list = self.generate_ngrams(self.wordcount)
            self.pos_ngram_list = self.generate_posngrams(self.wordcount)
            self.mixedgram_list = self.generate_mixed_ngrams(self.wordcount)
        else:
            self.ngram_list = self.generate_ngrams()
            self.pos_ngram_list = self.generate_posngrams()
            self.mixedgram_list = self.generate_mixed_ngrams()

        self.assign_ngrams()
        self.assign_pos_ngrams()
        self.assign_mix_ngrams()

        self.funcwords = 0
        for word in self.t_sent_upper:
            if len(word) >1:
                if not word.isupper():
                    if word.lower() in stop_wordlist:
                        self.funcwords += 1
            else:
                if word.lower() in stop_wordlist:
                    self.funcwords += 1
        self.contentwords = self.wordcount - self.funcwords
        self.generate_helper_dict()

    def generate_helper_dict(self):
        self.h_dict["unigram"] = self.unigram
        self.h_dict["bigram"] = self.bigram
        self.h_dict["trigram"] = self.trigram
        self.h_dict["quadgram"] = self.quadgram

        self.h_dict["unigram_p"] = self.unigram_p
        self.h_dict["bigram_p"] = self.bigram_p
        self.h_dict["trigram_p"] = self.trigram_p
        self.h_dict["quadgram_p"] = self.quadgram_p

        self.h_dict["unigram_m"] = self.unigram_m
        self.h_dict["bigram_m"] = self.bigram_m
        self.h_dict["trigram_m"] = self.trigram_m
        self.h_dict["quadgram_m"] = self.quadgram_m

    def assign_ngrams(self):
        if self.wordcount >= 4:
            self.quadgram = self.ngram_list[3]
        else:
            self.quadgram = []
        if self.wordcount >=3:
            self.trigram = self.ngram_list[2]
        else:
            self.trigram = []
        if self.wordcount>=2:
            self.bigram = self.ngram_list[1]
        else:
            self.bigram = []
        self.unigram = self.ngram_list[0]

    def assign_pos_ngrams(self):
        if self.wordcount >= 4:
            self.quadgram_p = self.pos_ngram_list[3]
        else:
            self.quadgram_p = []
        if self.wordcount >=3:
            self.trigram_p = self.pos_ngram_list[2]
        else:
            self.trigram_p = []
        if self.wordcount>=2:
            self.bigram_p = self.pos_ngram_list[1]
        else:
            self.bigram_p = []
        self.unigram_p = self.pos_ngram_list[0]

    def assign_mix_ngrams(self):
        if self.wordcount >= 4:
            self.quadgram_m = self.mixedgram_list[3]
        else:
            self.quadgram_m = []
        if self.wordcount >=3:
            self.trigram_m = self.mixedgram_list[2]
        else:
            self.trigram_m = []
        if self.wordcount>=2:
            self.bigram_m = self.mixedgram_list[1]
        else:
            self.bigram_m = []
        self.unigram_m = self.mixedgram_list[0]

    def generate_ngrams(self, n=4):
        ngram_list = []
        for i in range(1, n + 1):
            ngram_list.append(list(zip(*[self.t_sent[j:] for j in range(i)])))
        return ngram_list

    def generate_posngrams(self,n=4):
        ngram_list = []
        for i in range(1, n + 1):
            ngram_list.append(list(zip(*[self.pos_list[j:] for j in range(i)])))
        return ngram_list

    def generate_mixed_ngrams(self,n=4):
        ngram_list = []
        for i in range(1, n + 1):
            ngram_list.append(list(zip(*[self.pos[j:] for j in range(i)])))
        return ngram_list

    def print_sentfeatures(self):
        print self.wordcount
        print self.contentwords
        print self.t_sent
        #print self.punc_count
        print self.pos
        print self.bigram


if __name__ == '__main__':
    pass