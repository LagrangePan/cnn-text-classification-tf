# -*- encoding: utf-8 -*-
import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

import data_helpers

# 参数配置
# ==================================================

# 数据加载相关参数
tf.flags.DEFINE_string("positive_data_file", "./data/data_processed/test/ctrip_reviews.pos",
                       "Data source for the positive data")
tf.flags.DEFINE_string("negative_data_file", "./data/data_processed/test/ctrip_reviews.neg",
                       "Data source for the negative data")

# 模型评估相关参数
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("eval_model_dir", "./runs", "Model directory from training run")

# 其他参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# 读取参数
FLAGS = tf.flags.FLAGS

# 打印当前使用参数
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# 关闭警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 加载测试数据
x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# 最佳结果
best_test_accuracy = 0.0
best_model_dir = None

# 遍历所有模型，对比在测试集上的准确度
eval_model_dir = FLAGS.eval_model_dir
model_folders = os.listdir(eval_model_dir)
for model_folder in model_folders:
    print('Evaluating model: {}'.format(model_folder))
    model_dir = os.path.join(eval_model_dir, model_folder)
    # 加载词典
    vocab_path = os.path.join(model_dir, "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    # 评估模型
    # ==================================================
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
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
            input_y = graph.get_operation_by_name("input_y").outputs[0]

            # 获取准确度运算操作
            accuracy_op = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

            # 生成测试数据迭代器
            batches = data_helpers.batch_iter(list(zip(x_test, y_test)), FLAGS.batch_size)

            # 统计准确度
            all_accuracy = []
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                batch_accuracy = sess.run(accuracy_op, {input_x: x_batch, input_y: y_batch})
                all_accuracy.append(batch_accuracy)

            accuracy = np.mean(all_accuracy)
            print("Model accuracy: {:.2%}\n".format(accuracy))

            if accuracy > best_test_accuracy:
                best_test_accuracy = accuracy
                best_model_dir = model_dir

print("Best model in dir: {}".format(best_model_dir))
print("Best accuracy: {:.2%}".format(best_test_accuracy))

# 记录最佳模型
best_model_recorder = os.path.join(eval_model_dir, "best_model.p")
pickle.dump((best_model_dir, best_test_accuracy), open(best_model_recorder, "wb"))
print("\nModel saved to {}".format(best_model_recorder))
