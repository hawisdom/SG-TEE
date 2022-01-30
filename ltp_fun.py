import os
from pyltp import Postagger,NamedEntityRecognizer


LTP_DIR = "./model/ltp"

class LTP_Parse:
    def __init__(self):
        self.postagger = Postagger()
        pos_model_path = os.path.join(LTP_DIR, "pos.model")
        self.postagger.load_with_lexicon(pos_model_path, LTP_DIR + '/user_postag.txt')
        # self.postagger.load(os.path.join(LTP_DIR, "pos.model"))

        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(LTP_DIR, "ner.model"))

    def model_release(self):
        self.postagger.release()
        self.recognizer.release()

def get_ltp_info(ltp_model,sentence):
    words_sour = list(ltp_model.segmentor.segment(sentence))  # 分词

    words = ltp_model.forcesegmentor.merge(sentence, words_sour)  # 强制分词以后的结果
    postags = list(ltp_model.postagger.postag(words))  # 词性标注
    arcs = list(ltp_model.parser.parse(words, postags))  # 句法依存分析

    return words,postags,arcs

