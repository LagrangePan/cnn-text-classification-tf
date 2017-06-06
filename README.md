基于论文《[Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)》在Tensorflow上实现的中文文本分类

## 特点
- 中文文本分类
- 可选使用预训练的word2vec模型（二进制文件）
- 模型中实现了嵌入层双通道（论文中的“CNN-multichannel”）
- 模型中应用了[Highway Network](https://arxiv.org/abs/1505.00387)
- 模型中应用了Early Stopping
- 提供交互模式

## 依赖

- Python 3
- Tensorflow r1.1
- NumPy
- [Gensim](https://radimrehurek.com/gensim/)
- [scikit-learn](http://scikit-learn.org/stable/index.html)
- [jieba](https://pypi.python.org/pypi/jieba/)
- [tqdm](https://pypi.python.org/pypi/tqdm)

## 数据处理
```bash
./process_data.py
```
对原始数据进行预处理

## 模型训练

查看参数:

```bash
./train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --dev_sample_percentage DEV_SAMPLE_PERCENTAGE
                        Percentage of the training data to use for validation
  --positive_data_file POSITIVE_DATA_FILE
                        Data source for the positive data
  --negative_data_file NEGATIVE_DATA_FILE
                        Data source for the negative data
  --w2v_binary_file W2V_BINARY_FILE
                        Word2vec pre-trained binary file
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularization lambda (default: 0.0)
  --multichannel [MULTICHANNEL]
                        Use two sets of word vectors (default: False)
  --nomultichannel
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 200)
  --early_stopping EARLY_STOPPING
                        Early stopping rounds (default: 10)
  --allow_soft_placement [ALLOW_SOFT_PLACEMENT]
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement [LOG_DEVICE_PLACEMENT]
                        Log placement of ops on devices
  --nolog_device_placement

```

训练:

```bash
./train.py
```

## 评价模型
```bash
./eval.py
```
在已训练的模型中挑选出在测试集中表现最好的


## 交互模式
```bash
./interactive.py
```
读取在测试集上表现最好的模型进行交互式预测<br>
效果图<br>
![interactive demo](https://github.com/lianwj/cnn-text-classification-tf/blob/master/screenshots/interactive_demo.png)


## 参考论文
- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
- [Highway Networks](https://arxiv.org/abs/1505.00387)
