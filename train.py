# -*- encoding: utf-8 -*-
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn

import data_helpers
from text_cnn import TextCNN

# 参数配置
# ==================================================

# 数据加载相关参数
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/data_processed/train/ctrip_reviews.pos", "Data source for the positive data")
tf.flags.DEFINE_string("negative_data_file", "./data/data_processed/train/ctrip_reviews.neg", "Data source for the negative data")
tf.flags.DEFINE_string("w2v_binary_file", None, "Word2vec pre-trained binary file")
# tf.flags.DEFINE_string("w2v_binary_file", "./data/word2vec/reviews_vectors_100.bin", "Word2vec pre-trained binary file")

# 模型超参
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_boolean("multichannel", False, "Use two sets of word vectors (default: False)")  # 建议与预训练word2vec一起使用

# 训练相关参数
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("early_stopping", 10, "Early stopping rounds (default: 10)")

# 其他参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# 打印当前使用参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
parameters = sorted(FLAGS.__flags.items())
print("\nParameters:")
for attr, value in parameters:
    print("{}={}".format(attr.upper(), value))
print("")

# 数据准备
# ==================================================

# 加载数据
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# 构建词典
max_sentence_length = max([len(x.split()) for x in x_text])  # 最长的句子的长度
vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# 构建词向量矩阵
embedding_matrix = data_helpers.build_embedding_matrix(vocab_processor, FLAGS.embedding_dim, FLAGS.w2v_binary_file)

# 划分训练/验证数据集
x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=FLAGS.dev_sample_percentage)

# 训练
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
sess = tf.Session(config=session_conf)
with sess.as_default():
    cnn = TextCNN(
        sequence_length=x.shape[1],
        num_classes=y.shape[1],
        embedding_matrix=embedding_matrix,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        l2_reg_lambda=FLAGS.l2_reg_lambda,
        multichannel=FLAGS.multichannel)

# 定义训练过程
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-3)
grads_and_vars = optimizer.compute_gradients(cnn.loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# 统计训练过程中的梯度值及稀疏性
grad_summaries = []
for g, v in grads_and_vars:
    if g is not None:
        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
grad_summaries_merged = tf.summary.merge(grad_summaries)

# 模型参数以及统计信息存放目录
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))

# 统计目标：Loss、准确度
loss_summary = tf.summary.scalar("loss", cnn.loss)
acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

# 统计训练过程
train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

# 统计验证过程
dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

# 模型参数保存路径
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables())

# 保存字典
vocab_processor.save(os.path.join(out_dir, "vocab"))

# 初始化参数
sess.run(tf.global_variables_initializer())


def train_step(x_batch, y_batch):
    """
    训练操作
    """
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    # time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    train_summary_writer.add_summary(summaries, step)


def dev_step(x_batch, y_batch):
    """
    验证操作
    """
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch
    }
    step, summaries, loss, accuracy = sess.run(
        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    dev_summary_writer.add_summary(summaries, step)

    return loss, accuracy


# 最佳验证结果
best_dev_accuracy = 0.0
best_dev_epoch = 0

for epoch in range(FLAGS.num_epochs):
    print('Epoch {}'.format(epoch))

    # 训练
    batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size)
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)

    # 验证
    print("\nEvaluation:")
    current_loss, current_accuracy = dev_step(x_dev, y_dev)
    print("")

    # 保存最佳结果
    if current_accuracy > best_dev_accuracy:
        best_dev_accuracy = current_accuracy
        best_dev_epoch = epoch
        path = saver.save(sess, checkpoint_prefix)
        print("Saved model to {}\n".format(path))

    # 如果验证集上准确度多次没有提升，则终止训练
    if epoch - best_dev_epoch > FLAGS.early_stopping:
        print("Stopped at epoch: {}".format(str(epoch)))
        print("Best accuracy: {:.2%}".format(best_dev_accuracy))

        # 记录当前模型最佳准确度及使用参数
        with open(os.path.join(out_dir, "accuracy_with_parameters.txt"), "w") as f:
            f.write("Accuracy: {}\n".format(str(best_dev_accuracy)))
            f.write("\nParameters:")
            for attr, value in parameters:
                f.write("\n{}={}".format(attr.upper(), value))
        break
