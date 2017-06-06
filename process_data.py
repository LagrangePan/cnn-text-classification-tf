# -*- encoding: utf-8 -*-
import os

import jieba

import data_helpers


def transform_data(source_file, target_file):
    """
    数据转换（分词+清洗）
    Args:
        source_file: 源文件
        target_file: 目标文件
    """
    with open(target_file, "w", encoding="utf-8") as tf:
        with open(source_file, "r", encoding="utf-8", errors="ignore") as sf:
            lines = sf.readlines()
            for line in lines:
                line = " ".join(jieba.cut(line))
                line = data_helpers.clean_str(line)
                tf.write(line)
                tf.write("\n")


if __name__ == '__main__':
    source_file_list = ["./data/data_origin/train/ctrip_reviews.pos.txt",
                        "./data/data_origin/train/ctrip_reviews.neg.txt",
                        "./data/data_origin/test/ctrip_reviews.pos.txt",
                        "./data/data_origin/test/ctrip_reviews.neg.txt"]

    target_file_list = ["./data/data_processed/train/ctrip_reviews.pos",
                        "./data/data_processed/train/ctrip_reviews.neg",
                        "./data/data_processed/test/ctrip_reviews.pos",
                        "./data/data_processed/test/ctrip_reviews.neg"]

    for source_file, target_file in zip(source_file_list, target_file_list):
        source_dir = os.path.dirname(source_file)
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)

        target_dir = os.path.dirname(target_file)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        transform_data(source_file, target_file)

    print("Data-set created!")
