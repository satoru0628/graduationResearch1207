from adjustText import adjust_text
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from IPython.core.pylabtools import figsize
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import smart_open
import pandas as pd
import os
import time
import mojimoji
import warnings
import logging
import re
from tqdm import tqdm
import requests
import MeCab
from termcolor import cprint

warnings.filterwarnings('ignore')


class GensimDoc2Vec:
    def __init__(self, base_dir):
        slothlib_path = \
            r"http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"
        slothlib_file = requests.get(slothlib_path).text
        slothlib_stopwords = slothlib_file.split("\r\n")
        self.base_dir = base_dir
        self.stopwords = slothlib_stopwords
        self.data_dir = os.path.join(base_dir, "data")
        self.formatted_data_dir = os.path.join(base_dir, "formatted")
        self.wakati_data_dir = os.path.join(base_dir, "wakati")
        self.test_dir = os.path.join(base_dir, "test")
        self.train_dir = os.path.join(base_dir, "train")
        self.assess_dir = os.path.join(base_dir, "assess")
        self.model_dir = os.path.join(base_dir, "model")
        self.image_dir = os.path.join(base_dir, "image")
        try:
            os.mkdir(self.model_dir)
            os.mkdir(self.assess_dir)
            os.mkdir(self.data_dir)
            os.mkdir(self.formatted_data_dir)
            os.mkdir(self.wakati_data_dir)
            os.mkdir(self.train_dir)
            os.mkdir(self.test_dir)
            os.mkdir(self.image_dir)
        except FileExistsError:
            pass

    def get_data(self, data_dir, keywords=None):
        os.chdir(data_dir)
        data_list = os.listdir(data_dir)
        if not keywords:
            keywords = [i.split("_")[0] for i in data_list]

        for author in keywords:
            cprint("著者{}の作品を.txtファイルにまとめています。".format(author), "blue")
            author_summarized_file = open(os.path.join(self.data_dir, "{}.txt".format(author)), "w", encoding="UTF-8")
            author_works = [i for i in data_list if i.split("_")[0] == author]
            for author_work in author_works:
                print("著者{}: 作品名{}がまとめられています。".format(author_work.split("_")[0], author_work.split("_")[1]))
                author_work_file = open(author_work, "r", encoding="Shift-JIS")
                author_summarized_file.write(author_work_file.read())
                author_work_file.close()
            author_summarized_file.close()

    def format_data(self):
        def format_line(base_string):
            """英字をすべて小文字へ, 半角を全角へ、#〜や@~を削除, URLを削除, 日付の削除"""
            small_string = base_string.lower()
            zen_string = mojimoji.zen_to_han(small_string, digit=False, kana=False)
            han_string = mojimoji.han_to_zen(zen_string, digit=False, ascii=False)
            formatted_string = "".join(filter(lambda c: ord(c) < 0x10000, han_string))
            patterns = [r"@\w*", r"#(\w+)", r"(http(s)?(:)?//[\w | /]*)"]
            for replace in ["'", '"', ';', '.', ',', '-', '!', '?', '=', "(", ")", "「", "」", "|", "『", "』"]:
                formatted_string = formatted_string.replace(replace, "")

            for pattern in [re.compile(i) for i in patterns]:
                formatted_string = re.sub(pattern, "", formatted_string)
            return formatted_string

        cprint("文書の正規化を始めます。", "yellow")
        os.chdir(self.data_dir)
        for file_name in tqdm(os.listdir(self.data_dir)):
            new_file_name = str(file_name.split(".")[0]) + '_formatted.txt'
            w_file = open(os.path.join(self.formatted_data_dir, new_file_name), "a")
            with open(file_name, "r", encoding="UTF-8") as r_file:
                base_line = r_file.readlines()
                for base_string in base_line:
                    f_string = format_line(base_string)
                    w_file.write(f_string)
                w_file.close()

    def wakati(self):
        cprint("データの分かち書きを始めます。", "yellow")
        os.chdir(self.formatted_data_dir)
        for file_name in tqdm(os.listdir(self.formatted_data_dir)):
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

    def sep_test_train(self):
        cprint("データを訓練データ、テストデータに分割します。", "yellow")
        os.chdir(self.wakati_data_dir)
        for file in tqdm(os.listdir(self.wakati_data_dir)):
            num_lines = sum([1 for line in open(file)])
            train_file_name = str(file.split(".")[0]) + '_train.txt'
            test_file_name = str(file.split(".")[0]) + '_test.txt'
            train_file = open(os.path.join(self.train_dir, train_file_name), "w")
            test_file = open(os.path.join(self.test_dir, test_file_name), "w")
            with open(file, "r") as r_file:
                data_list = r_file.readlines()
                for i, line in enumerate(data_list):
                    if i <= num_lines * 0.9:
                        train_file.write(line)
                    else:
                        test_file.write(line)
            train_file.close()
            test_file.close()

    def train_model(self, dm, vector_size, min_count, workers, negative, window, log=True):
        if log:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        start = time.time()
        train_corpus = [i for i in self.read_corpus()]
        tag_name = ["_".join(i.split("_")[0]) for i in os.listdir(self.wakati_data_dir)]
        cprint("モデルの学習を始めます。", "yellow")
        model = Doc2Vec(dm=dm, vector_size=vector_size, min_count=min_count, epochs=200,
                        workers=workers, negative=negative, window=window)
        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        os.chdir(self.model_dir)
        model.save("model_v{}_w{}".format(vector_size, window))
        elapsed_time = time.time() - start
        cprint("モデルの学習を終了します。学習時間: {}sec".format(elapsed_time), "yellow")
        return tag_name

    def assess_model(self, vector_size, window):
        start = time.time()
        cprint("モデルの評価を始めます。", "yellow")
        os.chdir(self.model_dir)
        model = Doc2Vec.load("model_v{}_w{}".format(vector_size, window))
        tag_name = [i.split("_")[0] for i in os.listdir(self.wakati_data_dir)]
        result = {i: 0 for i in tag_name}
        for doc_tag in tag_name:
            test_data_list = [i for i in self.read_test_corpus(doc_tag)]
            test_data_len = len(test_data_list)
            for test_data in test_data_list:
                inferred_vector = model.infer_vector(test_data)
                sims = model.docvecs.most_similar([inferred_vector], topn=1)
                if sims[0][0] == doc_tag:
                    result[doc_tag] += 1
            result[doc_tag] = result[doc_tag] / (test_data_len + 1)
        result_now = pd.DataFrame({"vector_size": model.vector_size, "epochs": model.epochs,
                                   "sample": model.sample, "window": model.window,
                                   "High_A": np.max(list(result.values())), "Low_A": np.min(list(result.values())),
                                   "M_A": np.mean(list(result.values())),}, index=[0])
        cprint("""
        Model
            vector_size:{}
            epochs     :{}
            sample     :{}
            window     :{}
        High Accuracy  : {}
        Low Accuracy   : {}
        Mean Accuracy  : {}
        """.format(model.vector_size, model.epochs, model.sample, model.window,
                   np.max(list(result.values())), np.min(list(result.values())), np.mean(list(result.values()))), "red")
        os.chdir(self.assess_dir)
        if "result_all.xlsx" in os.listdir(self.assess_dir):
            result_all = pd.read_excel("result_all.xlsx")
            result_all = pd.concat([result_all, result_now])
            result_all.to_excel("result_all.xlsx")
        else:
            result_now.to_excel("result_all.xlsx")
        elapsed_time = time.time() - start
        cprint("モデルの評価を終了します。学習時間: {}sec".format(elapsed_time), "yellow")

    def read_corpus(self):
        cprint("corpusを作成しています。", "yellow")
        os.chdir(self.train_dir)
        doc = " "
        for file_name in tqdm(os.listdir(self.train_dir)):
            tag_name = file_name.split("_")[0]
            with smart_open.smart_open(file_name, encoding="UTF-8") as f:
                for i, line in enumerate(f):
                    if not line:
                        continue
                    elif i % 10 == 0:
                        yield TaggedDocument(doc.split(" "), [tag_name])
                        doc = ""
                    else:
                        doc = doc + " " + line

    def read_test_corpus(self, test_name):
        doc = ""
        os.chdir(self.test_dir)
        file_name = "{}_formatted_wakati_test.txt".format(test_name)
        with open(file_name, "r") as f:
            for i, line in enumerate(f.readlines()):
                if not line:
                    continue
                elif i % 10 == 0:
                    yield doc.split(" ")
                    doc = ""
                else:
                    doc = doc + " " + line

    def plot_tsne(self, window, vector_size):
        cprint("モデルのt-NSEによる出力を始めます。", "yellow")
        os.chdir(self.model_dir)
        model = Doc2Vec.load("model_v{}_w{}".format(vector_size, window))
        figsize(15, 15)
        plt.rcParams['font.family'] = 'IPAexGothic'
        tag_name = [i.split("_")[0] for i in os.listdir(self.train_dir)]
        train_data = [model[tag] for tag in tag_name]
        reduced_data = TSNE(n_components=2, random_state=0).fit_transform(train_data)
        for i, tag in enumerate(tag_name):
            plt.scatter(reduced_data[i, 0], reduced_data[i, 1], color=cm.hsv(i / 160))
        texts = [plt.text(reduced_data[i, 0], reduced_data[i, 1], str(tag_name[i]), ha='center', va='center') for i in
                 range(len(reduced_data))]
        adjust_text(texts)
        plt.xticks(color="None")
        plt.yticks(color="None")
        os.chdir(self.image_dir)
        plt.savefig("image_tsne_w{}_v{}.png".format(window, vector_size))

    def plot_ghc(self, window, vector_size):
        cprint("モデルのt-NSEによる出力を始めます。", "yellow")
        os.chdir(self.model_dir)
        model = Doc2Vec.load("model_v{}_w{}".format(vector_size, window))
        tag_name = [i.split("_")[0] for i in os.listdir(self.train_dir)]
        train_data = [model[tag] for tag in tag_name]
        df = pd.DataFrame(train_data, index=tag_name)
        linkage_result = linkage(df, method="ward", metric="euclidean")
        threshold = 0.7 * np.max(linkage_result[:, 2])
        plt.figure(num=None, figsize=(10, 15), dpi=200, facecolor='w', edgecolor="k")
        dendrogram(linkage_result, labels=tag_name, color_threshold=threshold, orientation="right", leaf_font_size=10)
        plt.xticks(color="None")
        os.chdir(self.image_dir)
        plt.savefig("image_ghc_w{}_v{}.png".format(window, vector_size))

    @classmethod
    def train_all_parameters(cls, base_dir, data_dir, keywords, window_list, vector_size_list, log):
        inst = cls(base_dir=base_dir)
        inst.get_data(data_dir, keywords)
        inst.format_data()
        inst.wakati()
        inst.sep_test_train()
        for window in window_list:
            for vector_size in vector_size_list:
                cprint("""Now Training : 
                   window     | {}
                   vector_size| {}""".format(window, vector_size), "red")
                inst.train_model(dm=1, vector_size=vector_size, min_count=10, workers=5,
                                 negative=10, window=window, log=log)
                inst.assess_model(window=window, vector_size=vector_size)
                inst.plot_tsne(window=window, vector_size=vector_size)
                inst.plot_ghc(window=window, vector_size=vector_size)


if __name__ == '__main__':
    data_dir_tmp = '/home/tomoya/PycharmProjects/sotuken/graduationResearch1207/data/base_data'  # データを出力する場所を書く lionが記述する場所
    base_dir_tmp = '/home/tomoya/PycharmProjects/sotuken/graduationResearch1207/data/result'  # 元々のデータが存在する場所を書く　lionが記述する場所
    authors = ["芥川龍之介", "有島武郎", "石川啄木"]  # lionが記述する場所
    log = False
    window_list_tmp = [i for i in range(5, 20)]
    vector_size_tmp = [i for i in range(50, 500, 50)]
    GensimDoc2Vec.train_all_parameters(base_dir=base_dir_tmp,
                                       data_dir=data_dir_tmp,
                                       keywords=authors,
                                       window_list=window_list_tmp,
                                       vector_size_list=vector_size_tmp,
                                       log=log)
