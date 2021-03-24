import pandas as pd
import gensim
import numpy as np
from bert4keras.snippets import DataGenerator, sequence_padding
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score
from bert4keras.backend import keras, K

SEED=2020
num_classes = 19
vocabulary_size = 7000
maxlen=1024
batch_size = 128
# embedding_dim = 256 # 不预设词向量可以随便改
embedding_dim = 200 # 匹配固定词向量 改成200
num_filters = 512
filter_sizes = [3,4,5]
drop = 0.5
lr = 1e-4
epochs = 20

wordkeyvector = "tc/70000-small.txt"  # 词向量地址
news = "tc/article_features_train_raw.csv" # 预料地址

# 加载词向量模型
def w2v_model_preprocessing():
    # 导入模型
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(wordkeyvector, binary=False)
    word2idx = {"_PAD": 0} # 词->index
    vocab_list = [(k, w2v_model.wv[k]) for k, v in w2v_model.wv.vocab.items()]
    # 存储所有word2vec中所有词向量的数组，其中第一个全为0用于padding
    embeddings_matrix = np.zeros((len(w2v_model.wv.vocab.items())+1, w2v_model.vector_size))
    # 填充字典和矩阵
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i+1
        embeddings_matrix[i+1] = vocab_list[i][1]
    return w2v_model, word2idx, embeddings_matrix
w2v_model, word2idx, embeddings_matrix = w2v_model_preprocessing()

# 加载预料数据
df = pd.read_csv(news)
df = df.dropna(axis=0, how="any")
def get_words_index(data, word_index):
    new_txt = []
    for word in data:
        try:
            new_txt.append(word_index[word])
        except:
            new_txt.append(0)
    return new_txt
lable_index = {'艺术':0, '文学':1, '哲学':2, '通信':3, '能源':4, '历史':5,
               '矿藏':6, '空间':7, '教育':8, '交通':9, '计算机':10, '环境':11,
               '电子':12, '农业':13, '体育':14, '时政':15, '医疗':16, '经济':17, '法律':18}
df["label"] =df["label"].apply(lambda x:lable_index[x])
df["text"] = df["words"].apply(lambda x:get_words_index(x.split(), word2idx))

# 拆分训练集测试集
from sklearn.model_selection import train_test_split
df_train, df_valid = train_test_split(df, test_size=0.2, random_state=SEED)
def load_data(df):
    D = list()
    for _, row in df.iterrows():
        text = row["text"]
        label = row["label"]
        D.append((text, int(label)))
    return D
train_data = load_data(df_train)
valid_data = load_data(df_valid)

# 构建模型
class Self_Attention(Layer):
    def __init__(self, dk, **kwargs):
        self.dk = dk
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[-1], self.dk),  # 就是生成qkv的工具矩阵
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        # 这里输入的x一定是三个维度
        x = K.reshape(x, shape=(-1, 1, K.shape(x)[-1]))  # N*1*512
        q = K.dot(x, self.kernel[0])  # N*1*512 dot* 512*dk = N*1*dk
        k = K.dot(x, self.kernel[1])
        v = K.dot(x, self.kernel[2])

        k = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))
        k = k / (self.dk ** 0.5)
        k = K.softmax(k)
        v = K.batch_dot(k, v)
        v = K.squeeze(v, axis=1)
        return v

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dk)
# 输入
inputs = Input(shape=(maxlen,), dtype="int32")

# 嵌入层
embedding = Embedding(
#     input_dim=vocabulary_size, # 词典size
    input_dim=len(embeddings_matrix), # 输入因为引入词向量所以这里输入维度改变成词向量size
    output_dim=embedding_dim, # 词向量size
    input_length=maxlen, # 输入size
    weights=[embeddings_matrix],  # 为了引入词向量
    trainable=False # 引入词向量不训练
)(inputs) # 输入

reshape = Reshape((maxlen, embedding_dim, 1))(embedding) # 加一个维度

# 卷积层
conv_0 = Conv2D(
    num_filters, # 输出size
    kernel_size=(filter_sizes[0], embedding_dim), # 卷积核宽高，宽=卷积核数，高=词向量size
    padding="valid",
    kernel_initializer="normal",
    activation="relu"
)(reshape)
conv_1 = Conv2D(
    num_filters, # 输出size
    kernel_size=(filter_sizes[1], embedding_dim), # 卷积核宽高，宽=卷积核数，高=词向量size
    padding="valid",
    kernel_initializer="normal",
    activation="relu"
)(reshape)
conv_2 = Conv2D(
    num_filters, # 输出size
    kernel_size=(filter_sizes[2], embedding_dim), # 卷积核宽高，宽=卷积核数，高=词向量size
    padding="valid",
    kernel_initializer="normal",
    activation="relu"
)(reshape)
# 池化
maxpool_0 = MaxPool2D(
    pool_size = (maxlen-filter_sizes[0]+1, 1),
    strides = (1,1),
    padding="valid"
)(conv_0)
maxpool_1 = MaxPool2D(
    pool_size = (maxlen-filter_sizes[1]+1, 1),
    strides = (1,1),
    padding="valid"
)(conv_1)
maxpool_2 = MaxPool2D(
    pool_size = (maxlen-filter_sizes[2]+1, 1),
    strides = (1,1),
    padding="valid"
)(conv_2)

# 输出层
concatenated_tensor = Concatenate(axis=1)([maxpool_0,maxpool_1,maxpool_2])
flatten = Flatten()(concatenated_tensor)
output = Dropout(drop)(flatten)
output = Self_Attention(128)(output) # 128是attention隐藏层单元数
output = Dropout(drop)(output)
output = Dense(units=num_classes, activation="softmax")(output)
model = Model(inputs=inputs, outputs=output)
model.compile(
    optimizer=Adam(lr=lr),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
# model.summary()

# callback
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score
import numpy as np
class Evaluator(Callback):
    def __init__(self):
        super().__init__()
        self.best_val_f1 = 0
    def evaluate(self):
        y_true, y_pred = list(), list()
        for x, y in valid_generator:
            y_true.append(y)
            y_pred.append(self.model.predict(x).argmax(axis=1))
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        return f1
    def on_epoch_end(self, epoch, logs=None):
        val_f1 = self.evaluate()
        if val_f1>self.best_val_f1:
            self.best_val_f1 = val_f1
        logs["val_f1"] = val_f1
        print(f"val_f1:{val_f1:.5f}, best_val_f1:{self.best_val_f1:.5f}")

callbacks = [
    Evaluator(),
    EarlyStopping(
        monitor = "val_loss",
        patience = 1,
        verbose = 1
    ),
    ModelCheckpoint(
        "textcnn_best_model.weights",
        monitor="val_f1",
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
        mode="max"
    ),
]

model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    callbacks=callbacks,
    validation_data=valid_generator.forfit(),
    validation_steps=len(valid_generator)
)


# 训练
# model.load_weights("best_model.weights")
model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
#     callbacks=callbacks,
    validation_data=valid_generator.forfit(),
    validation_steps=len(valid_generator)
)
model.save_weights('best_model.weights')

def predictdemo():
    df_test = df_valid[:100]
    df_test["label"] = 0
    test_data = load_data(df_test)
    test_generator = data_generator(test_data, batch_size)
    result = model.predict_generator(test_generator.forfit(), steps=len(test_generator))
    result = result.argmax(axis=1)
    df_test["label"] = result
    # df_test.to_csv("submission.csv", index=False, columns=["label"])
    return df_test