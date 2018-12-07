from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import smart_open
import os
import logging
from pprint import pprint
import mojimoji
import emoji
import re
from tqdm import tqdm
import requests
import MeCab


class GensimDoc2Vec:
    def __init__(self, base_dir):
        slothlib_path = \
            r"http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"
        slothlib_file = requests.get(slothlib_path).text
        slothlib_stopwords = slothlib_file.split("\r\n")
        self.stopwords = slothlib_stopwords
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.format_data_dir = os.path.join(base_dir, "format")
        self.wakati_data_dir = os.path.join(base_dir, "wakati")
        self.test_data_dir = os.path.join(base_dir, "test")
        self.train_data_dir = os.path.join(base_dir, "train")
        self.model_dir = os.path.join(base_dir, "model")
        self.image_dir = os.path.join(base_dir, "image")
        self.result_dir = os.path.join(base_dir, "result")

    def get_data(self, data_dir):
        pass

    def format_data(self):
        os.chdir(self.base_dir)
        os.mkdir(self.format_data_dir)

        def format_line(base_string):
            """英字をすべて小文字へ, 半角を全角へ、#〜や@~を削除, URLを削除, 日付の削除"""
            small_string = base_string.lower()
            zen_string = mojimoji.zen_to_han(small_string, digit=False, kana=False)
            han_string = mojimoji.han_to_zen(zen_string, digit=False, ascii=False)
            formatted_string = "".join(c for c in han_string if c not in emoji.UNICODE_EMOJI)
            patterns = [r"@\w*", r"#(\w+)", r"(http(s)?(:)?//[\w | /]*)"]
            for replace in ["'", '"', ';', '.', ',', '-', '!', '?', '=', "(", ")", "「", "」", "|", "『", "』"]:
                formatted_string = formatted_string.replace(replace, "")

            for pattern in [re.compile(i) for i in patterns]:
                formatted_string = re.sub(pattern, "", formatted_string)
            return formatted_string

        print("文書の正規化を始めます。")
        os.chdir(self.data_dir)
        data_list = [i for i in os.listdir(self.data_dir) if ("txt" in i.split(".") and
                                                              "formatted" not in i.split("_"))]
        for file_name in tqdm(data_list):
            new_file_name = str(file_name.split(".")[0]) + '_formatted.txt'
            w_file = open(os.path.join(new_file_name, new_file_name), "a")
            with open(file_name, "r", encoding="UTF-8") as r_file:
                base_line = r_file.readlines()
                for base_string in base_line:
                    f_string = format_line(base_string)
                    w_file.write(f_string)
                w_file.close()
        print("文書の正規化を終了します。")

    def wakati(self):
        print("データの分かち書きを始めます。")
        os.chdir(self.format_data_dir)
        data_list = [i for i in os.listdir(self.format_data_dir) if ("txt" in i.split(".") and
                                                              "formatted.txt" in i.split("_"))]
        for file_name in tqdm(data_list):
            new_file_name = str(file_name.split(".")[0]) + '_wakati.txt'
            w_file = open(os.path.join(self.wakati_data_dir, new_file_name), "w")
            with open(file_name, "r") as r_file:
                tagger = MeCab.Tagger('-F\s%f[6] -U\s%m -E\\n')
                r_line = r_file.readline()
                while r_line:
                    result = tagger.parse(r_line)
                    rm_result = [i for i in result[1:].split(" ") if not i in self.stopwords]
                    w_file.write(" ".join(rm_result))
                    r_line = r_file.readline()
            w_file.close()

    def train_model(self, dm, vector_size, min_count, workers, negative, sample, window, log=True):
        if log:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        train_corpus = list(self.read_corpus())
        print("corpusの作成を終了します。")
        print("モデルの学習を始めます。")
        data_list = [i for i in os.listdir(self.wakati_data_dir) if "wakati.txt" in i.split("_")]
        tag_name = []
        for file_name in data_list:
            tag_name.append("_".join([str(i) for i in file_name.split("_")
                                 if (i not in ["normed", "wakati.txt", "wakati", "formatted"])]))
        model = Doc2Vec(dm=dm, vector_size=vector_size, min_count=min_count, epochs=100,
                        workers=workers, negative=negative, sample=sample, window=window)
        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(os.path.join(self.model_dir, "model_v{}_w{}.pk".format(vector_size, window)))

    def assess_model(self):
        pass

    def plot_t_NSE(self):
        pass

    def read_corpus(self):
        print("corpusを作成しています。")
        os.chdir(self.data_dir)
        data_list = [i for i in os.listdir(self.data_dir) if "wakati.txt" in i.split("_")]
        for file_name in data_list:
            tag_name = "_".join([str(i) for i in file_name.split("_")
                                 if (i not in ["normed", "wakati.txt", "wakati", "formatted"])])
            print(tag_name)
            with smart_open.smart_open(file_name, encoding="UTF-8") as f:
                for i, line in enumerate(f):
                    yield TaggedDocument(line, tag_name)

    @classmethod
    def learn_all(cls):
        pass


if __name__ == '__main__':
    pass