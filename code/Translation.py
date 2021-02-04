import os

import jieba
from nltk import CoreNLPParser
from nltk.parse import stanford
from nltk.internals import find_jars_within_path
from stanfordcorenlp import StanfordCoreNLP


class Translation:
    parser = CoreNLPParser('http://127.0.0.1:9010')
  
    def __init__(self, source, trans):
        self.source = source
        self.trans = trans
        self.seg_list = jieba.lcut(trans)
        seg_str = " ".join(self.seg_list)
        sent = Translation.parser.parse(self.seg_list)
        for line in sent:
            self.nlp_tree = line


    def get_translation(self):
        return self.source, self.trans, self.nlp_tree, self.seg_list
