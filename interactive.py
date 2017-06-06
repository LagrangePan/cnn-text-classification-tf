import os
import pickle

import jieba
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

import data_helpers

# 模型参数
tf.flags.DEFINE_string("best_model_recorder", "./runs/best_model.p", "File recorded the best model")

# 其他参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# 读取参数
FLAGS = tf.flags.FLAGS

# 关闭警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 手动初始化分词
jieba.initialize()

# 加载最佳模型路径
best_model_dir, best_accuracy_history = pickle.load(open(FLAGS.best_model_recorder, "rb"))
checkpoint_dir = os.path.join(best_model_dir, "checkpoints")
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

# 加载词典
vocab_path = os.path.join(best_model_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 加载保存的模型
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 获取输入、输出占位符
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        max_num_words = int(input_x.shape[1])

        # 获取分类得分以及预测值运算操作
        scores_op = graph.get_operation_by_name("output/scores").outputs[0]
        predictions_op = graph.get_operation_by_name("output/predictions").outputs[0]

        print("\n模型加载完毕\n")
        print("请根据提示输入对酒店的评价，系统根据评价进行分类（好评/差评）\n")
        print("提示：")
        print("　　1. 当前系统只能处理最多{:d}个汉语单词，超出部分将被系统忽略".format(max_num_words))
        print("　　2. 输入exit()可退出\n")

        while True:
            # 获取输入
            input_raw = input("\n请输入您对酒店的评价：\n")

            if input_raw == "exit()":
                break

            # 分词+清洗+转换
            input_cut = " ".join(jieba.cut(input_raw))
            input_cleaned = data_helpers.clean_str(input_cut)
            input_transformed = np.array(list(vocab_processor.transform([input_cleaned])))

            # 预测
            scores, predictions = sess.run((scores_op, predictions_op), {input_x: input_transformed})
            score_pos = scores[0][1]
            score_neg = scores[0][0]
            predictions_readable = "好评" if predictions[0] == 1 else "差评"

            print("\n正分类得分：{:.2f}".format(score_pos))
            print("负分类得分：{:.2f}".format(score_neg))
            print("预测结果：{}\n".format(predictions_readable))
