# -*- encoding: utf-8 -*-
import itertools
import re
from collections import Counter, defaultdict

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm


def normalize_punctuation(text):
    """
    标点符号标准化

    Args:
        text: 待处理的文本

    Returns:
        将中文标点转换为英文标点后的文本
    """
    # 中文标点
    cn_pun = [["	"],
              ["﹗"],
              ["“", "゛", "〃", "′"],
              ["”"],
              ["´", "‘", "’"],
              ["；", "﹔"],
              ["《", "〈", "＜"],
              ["》", "〉", "＞"],
              ["﹑"],
              ["【", "『", "〔", "﹝", "｢", "﹁"],
              ["】", "』", "〕", "﹞", "｣", "﹂"],
              ["（", "「"],
              ["）", "」"],
              ["﹖"],
              ["︰", "﹕"],
              ["・", "．", "·", "‧", "°"],
              ["●", "○", "▲", "◎", "◇", "■", "□", "※", "◆"],
              ["〜", "～", "∼"],
              ["︱", "│", "┼"],
              ["╱"],
              ["╲"],
              ["—", "ー", "―", "‐", "−", "─", "﹣", "–", "ㄧ"]]

    # 英文标点
    en_pun = [" ", "！", "\"", "\"", "'", ";", "<", ">", "、", "[", "]",
              "(", ")", "?", "：", "･", "•", "~", "|", "/", "\\", "-"]

    # 构建中英文标点替换映射
    replace_mapping = {}
    for i in range(len(cn_pun)):
        for j in range(len(cn_pun[i])):
            replace_mapping[cn_pun[i][j]] = en_pun[i]

    # 根据替换映射替换中文标点
    normalized_text = re.sub("|".join(re.escape(key) for key in replace_mapping.keys()),
                             lambda k: replace_mapping[k.group(0)], text)

    return normalized_text


def full2half(char):
    """
    全角转半角

    Args:
        char: 待转换的字符
    Returns:
        半角字符
    """
    unicode = ord(char)
    # 全角空格直接转半角空格
    if unicode == 0x3000:
        unicode = 0x0020
    else:
        unicode -= 0xfee0
    # 转换后不是半角字符返回原来的字符
    if unicode < 0x0020 or unicode > 0x7e:
        return char

    return chr(unicode)


def clean_str(string):
    """
    数据清洗

    Args:
        string: 待清洗字符串

    Returns:
        清洗后的字符串
    """
    # 标点符号标准化
    string = normalize_punctuation(string)

    # 全角字符转半角
    string = "".join([full2half(char) for char in list(string)])

    # 清洗文本中各种符号等不必要数据
    string = re.sub(r"# \d+", " ", string)
    string = re.sub(r"#", " ", string)
    string = re.sub(r"&\.*;", " ", string)
    string = re.sub(r"& gt ", " ", string)
    string = re.sub(r"& lt ", " ", string)
    string = re.sub(r"& quot ", " ", string)
    string = re.sub(r"& amp ", " ", string)
    string = re.sub(r"&.*?\d+", " ", string)
    string = re.sub(r"[，,、`@；;。\.･（）\(\)\[\]\{\}\<\>\"\'\~\!\?\-\:\\\/&\$%\^\*\_\=\+\|]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def data_filter(data, keep_percentage=0.8):
    """
    过滤数据，保留句子长度出现频率较高的数据

    Args:
        data: 原始数据
        percentage 保留数据百分比

    Returns:
        句子长度出现频率较高的数据
    """
    # 统计句子长度出现频率，Key为句子长度，Value为句子长度出现的次数
    length_count = defaultdict(int)
    for sentence in data:
        length_count[len(sentence.split())] += 1

    # 根据句子长度出现次数排序
    length_count_list = sorted(length_count.items(), key=lambda d: d[1], reverse=True)

    data_size = len(data)
    accepted_size = 0
    accept_length = []
    for item in length_count_list:
        # item: (句子长度, 句子出现次数)
        accept_length.append(item[0])
        accepted_size += item[1]
        if accepted_size / data_size > keep_percentage:
            break

    accepted_data = [sentence for sentence in data if len(sentence.split()) in accept_length]
    return accepted_data


def load_data_and_labels(positive_data_file, negative_data_file, keep_percentage=0.8):
    """
    加载数据及其标签
    标签以one-hot形式保存
    [0, 1] 表示正分类
    [1, 0] 表示负分类
    """
    # 从文件中读取数据并分词清洗过滤
    positive_examples = list(open(positive_data_file, "r", encoding="utf-8").readlines())
    # positive_examples = [clean_str(" ".join(jieba.cut(string))) for string in positive_examples]
    positive_examples = data_filter(positive_examples, keep_percentage)
    negative_examples = list(open(negative_data_file, "r", encoding="utf-8").readlines())
    # negative_examples = [clean_str(" ".join(jieba.cut(string))) for string in negative_examples]
    negative_examples = data_filter(negative_examples, keep_percentage)

    x_text = positive_examples + negative_examples

    # 生成one-hot形式标签
    # [0, 1] 表示正分类
    # [1, 0] 表示负分类
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return (x_text, y)


def build_vocabulary(data):
    # 构建词典
    word_counts = Counter(itertools.chain(*data))

    # 构建“词语-索引”映射
    vocabulary = {x: i for i, x in enumerate(sorted(word_counts.keys()))}

    return vocabulary


def build_embedding_matrix(vocab_processor, embedding_dim, w2v_binary_file=None):
    """
    构建词向量矩阵

    Args:
        vocab_processor: 提取词袋后的分词器（即针对训练数据执行fit操作后的VocabularyProcessor实例）
        embedding_dim: 词向量维度大小
        w2v_binary_file: 预训练的word2vec二进制文件

    Returns:
        词向量矩阵
    """

    print("Building word embeddings...")
    vocabulary = vocab_processor.vocabulary_._reverse_mapping
    word_idx_map = vocab_processor.vocabulary_._mapping

    if w2v_binary_file is not None:
        print("Loading word2vec binary file...")
        w2v_model = KeyedVectors.load_word2vec_format(w2v_binary_file, binary=True)

        embedding_dim = w2v_model.vector_size
        embeddings = np.random.uniform(-1.0, 1.0, [len(vocabulary), embedding_dim])
        # 预训练词在词向量矩阵中的值
        for word, idx in word_idx_map.items():
            if word in w2v_model.vocab:
                embeddings[idx] = w2v_model.word_vec(word)
    else:
        embeddings = np.random.uniform(-1.0, 1.0, [len(vocabulary), embedding_dim])

    print("Word embeddings built.")

    return embeddings


def batch_iter(data, batch_size):
    """
    生成批量数据的迭代器
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for batch_num in tqdm(range(num_batches_per_epoch)):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]
