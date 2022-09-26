from cgi import print_environ
from unicodedata import name
import tensorflow  as tf


class IR_Block(tf.keras.layers.Layer):

    def __init__(self, filters=64, strides=(1, 1)):
        super(IR_Block, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.PReLU() 
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', strides=strides)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', strides=strides)
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if x.shape != inputs.shape:
            res = self.conv3(inputs)
            res = self.bn3(res, training=training)
            x += res
            
        # x = self.act(x)
        return x


class hiendvModel(tf.keras.Model):
    def __init__(self, Block=IR_Block, layers=(3, 4, 14, 3), include_top=True, embedding_size=512, dropout_rate=0):
        super(hiendvModel, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.PReLU()
        # self.maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')
        self.blocks1 = tf.keras.Sequential(
            [Block(filters=64, strides=(1, 1)) for _ in range(layers[0])])
        self.blocks2 = tf.keras.Sequential(
            [Block(filters=128, strides=(2, 2) if i == 0 else (1, 1)) for i in range(layers[1])])
        self.blocks3 = tf.keras.Sequential(
            [Block(filters=256, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[2])])
        self.blocks4 = tf.keras.Sequential(
            [Block(filters=512, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[3])])
        self.bn2 = tf.keras.layers.BatchNormalization()
        # self.globalpool = tf.keras.layers.GlobalAveragePooling2D()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = None
        if include_top:
            self.dense = tf.keras.layers.Dense(embedding_size)
        self.bn3 = tf.keras.layers.BatchNormalization(scale=False)

    def call(self, inputs, training=False, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.act(x)
        # x = self.maxpool(x)
        x = self.blocks1(x, training=training)
        x = self.blocks2(x, training=training)
        x = self.blocks3(x, training=training)
        x = self.blocks4(x, training=training)
        x = self.bn2(x, training=training)
        x = self.dropout(x, training=training)
        # x = self.globalpool(x)
        if self.dense is not None:
            x = self.dense(x)
        x = self.bn3(x, training=training)
        return x


class hiendvModel_50(hiendvModel):
    def __init__(self, include_top=True, embedding_size=512):
        super(hiendvModel_50, self).__init__(Block=IR_Block, layers=(3, 4, 14, 3), include_top=include_top,
                                           embedding_size=embedding_size)


