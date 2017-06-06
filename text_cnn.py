# -*- encoding: utf-8 -*-
import tensorflow as tf


class TextCNN(object):
    """
    文本分类CNN模型
    模型结构：
        - 嵌入层（Embedding layer）
        - 卷积层（Convolutional Layer）
        - 池化（Pooling）
        - Highway
        - 输出层
    """

    def __init__(self, sequence_length, num_classes, embedding_matrix, filter_sizes,
                 num_filters, l2_reg_lambda=0.0, multichannel=False):
        """
        初始化文本分类CNN模型实例

        Args:
            sequence_length: 句子（最大）长度
            num_classes: 目标分类数量
            embedding_matrix: 词向量矩阵
            filter_sizes: 卷积层过滤器大小列表
            num_filters: 卷积层过滤器个数
            l2_reg_lambda: L2正则项系数
            multichannel: 是否使用双嵌入层
        """

        # 输入、输出、Dropout占位符
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder_with_default(tf.constant(1.0), [], name="dropout_keep_prob")

        # 嵌入层（Embedding layer）
        with tf.variable_scope("embedding"):
            initializer = tf.constant_initializer(embedding_matrix)
            self.U = tf.get_variable(name="U", shape=embedding_matrix.shape, initializer=initializer,
                                     trainable=True)
            # self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.U, self.input_x)

            # conv2d input shape: [batch, in_height, in_width, in_channels]
            #   结合参考论文1中的Figure 1理解以下参数
            #   batch: 输入批量，相当于输入句子的个数
            #   in_height: 输入的“高度”，相当于句子的长度（词数量），对应论文中的n
            #   in_width: 输入的“宽度”，即词向量的维数，对应论文中的k
            #   in_channels: 输入的通道数，对应论文中的通道（channel）

            # 生成embedded_chars_expanded作为conv2d的输入数据（input）
            if multichannel:
                # 增加一个使用静态词向量矩阵的通道
                self.U_static = tf.get_variable(name="U_static", shape=embedding_matrix.shape, initializer=initializer,
                                                trainable=False)
                # self.W_static = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                #                             name="W_static", trainable=False)
                self.embedded_chars_static = tf.nn.embedding_lookup(self.U_static, self.input_x)
                self.embedded_chars_expanded = tf.stack([self.embedded_chars, self.embedded_chars_static], axis=3)
            else:
                # 直接扩展维度，满足conv2d的入参要求
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # 使用不同大小的过滤器进行卷积+池化，提取特征
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积层（Convolution Layer）
                embedding_size = int(embedding_matrix.shape[1])
                num_channels = int(self.embedded_chars_expanded.shape[3])
                filter_shape = [filter_size, embedding_size, num_channels, num_filters]
                # conv2d filter shape: [filter_height, filter_width, in_channels, output_channels]
                #   filter_height: 过滤器的高度，表次每次卷积单词的个数，对应论文中的h
                #   filter_width: 过滤器的宽度，等于词向量的维度大小，对应论文中的k
                #   in_channels: 输入数据的通道数，在这里可以理解为过滤器的“厚度”，对应论文中的通道（channel）
                #   output_channels: 在这里可以理解为卷积结果的“厚度”，比如使用10个过滤器就能卷积出10层“厚”的特征

                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # 使用relu激活
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 最大池化
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # 合并不同大小过滤器卷积+池化后的特征
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Add highway
        with tf.name_scope("highway"):
            W_T = tf.Variable(tf.truncated_normal([num_filters_total, num_filters_total], stddev=0.1),
                              name="weight_transform")
            b_T = tf.Variable(tf.constant(-1.0, shape=[num_filters_total]),
                              name="bias_transform")

            W = tf.Variable(tf.truncated_normal([num_filters_total, num_filters_total], stddev=0.1), name="weight")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="bias")

            T = tf.sigmoid(tf.matmul(self.h_drop, W_T) + b_T, name="transform_gate")
            H = tf.nn.relu(tf.matmul(self.h_drop, W) + b, name="activation")
            C = tf.subtract(1.0, T, name="carry_gate")

            self.highway = tf.add(tf.multiply(H, T), tf.multiply(self.h_drop, C), "y")

        # 输出
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
            # 各分类分数
            self.scores = tf.nn.xw_plus_b(self.highway, W, b, name="scores")
            # 取分数最大的分类
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # 计算平均交叉熵损失
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # 计算准确度
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
