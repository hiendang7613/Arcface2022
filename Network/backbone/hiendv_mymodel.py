
from cgi import print_environ
from unicodedata import name
import tensorflow  as tf


class IR_Block(tf.keras.layers.Layer):

    def __init__(self, filters=64, strides=(1, 1)):
        super(IR_Block, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.PReLU() 
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', strides=strides)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(filters//2, (3, 3), padding='same', strides=strides)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.act3 = tf.keras.layers.PReLU() 
        self.conv4 = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization()

        self.act5 = tf.keras.layers.PReLU()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
	       
        res = inputs
        if x.shape != inputs.shape:
            res = self.conv3(res)
            res = self.bn3(res, training=training)
            res = self.act3(res)
            res = self.conv4(res)
            res = self.bn4(res, training=training)
        x += res

        x = self.act5(x)
        return x


class hiendvModel(tf.keras.Model):
    def __init__(self, Block=IR_Block, layers=(3, 4, 14, 3), include_top=True, embedding_size=512, dropout_rate=0.2, input_shape=(160,160,3)):
        super(hiendvModel, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.PReLU()
        self.conv2 = tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.PReLU()
        self.maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')
        self.blocks1 = tf.keras.Sequential(
            [Block(filters=64, strides=(1, 1)) for _ in range(layers[0])])
        self.blocks2 = tf.keras.Sequential(
            [Block(filters=128, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[1])])
        self.blocks3 = tf.keras.Sequential(
            [Block(filters=256, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[2])])
        self.blocks4 = tf.keras.Sequential(
            [Block(filters=512, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[3])])
        self.globalpool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.act3 = tf.keras.layers.PReLU()
        kernel_size = int(input_shape[0]/32)
        self.reshape = tf.keras.layers.Reshape((kernel_size, kernel_size,512,1))
        self.conv3D = tf.keras.layers.Conv3D(1, (kernel_size, kernel_size, 1), strides=1, padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = None
        if include_top:
            self.dense = tf.keras.layers.Dense(embedding_size)
        self.bn3 = tf.keras.layers.BatchNormalization(scale=False)

    def call(self, inputs, training=False, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.act(x)

        res = x
        res = self.maxpool(res)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x+=res

        x = self.blocks1(x, training=training)
        x = self.blocks2(x, training=training)
        x = self.blocks3(x, training=training)
        x = self.blocks4(x, training=training)
        x = self.dropout(x, training=training)

        res = x
        res = self.globalpool(res)
        x = self.reshape(x)
        x = self.conv3D(x)
        x = self.flatten(x)
        x = self.act3(x)
        x = self.bn3(x, training=training)
        x+=res

        if self.dense is not None:
            x = self.dense(x)
        return x

class hiendvModel_50(hiendvModel):
    def __init__(self, include_top=True, embedding_size=512):
        super(hiendvModel_50, self).__init__(Block=IR_Block, layers=(3, 4, 14, 3), include_top=include_top, embedding_size=embedding_size)
