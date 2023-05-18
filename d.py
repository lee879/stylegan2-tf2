import tensorflow as tf
from tensorflow.keras.layers import Conv2D,LeakyReLU,AveragePooling2D,Dense,Layer
from tensorflow.keras import layers,Model,Sequential
class PixelNorm(Layer):  #这一层主要是进行标准化, tf自带的bn有很大的问题，
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs * tf.math.rsqrt(
            tf.math.mean(inputs * inputs, 1, keepdim=True) + 1e-8)

class block(layers.Layer):
    def __init__(self,fil,p = 1):
        super(block, self).__init__()
        self.res = Conv2D(fil, 1, kernel_initializer='he_uniform')
        self.conv1 = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_uniform')
        self.ac = LeakyReLU(0.2)
        self.conv2 = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_uniform')
        self.ac = LeakyReLU(0.2)
        self.down = AveragePooling2D(2)
        self.pn = PixelNorm()
        self.p = p

    def call(self, inputs, *args, **kwargs):
        x1 = self.res(inputs)
        x2 = self.conv1(inputs)
        x2 = self.ac(x2)
        x2 = self.conv2(x2)
        out = (layers.add([x2,x1]))
        if self.p == 1:
            out = self.down(out)
        out = self.pn(out)
        return out

class D(Model):
	# 残差鉴别器
    def __init__(self,scal=1):
        super(D, self).__init__()
        self.f = [int(64 * scal), int(128* scal), int(256* scal), int(512* scal), int(512* scal)]
        self.p = [1, 1, 1, 1, 1]
        self.conv = [block(fil) for fil in self.f]
        self.find = Sequential([
            layers.Flatten(),
            Dense(1)
        ])

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for i in range(len(self.f)):
            x = self.conv[i](x)
        return self.find(x)

# model = D()
# x = tf.random.normal([16,256,256,3])
# y = model(x)
# model.summary()
# print(y.shape)




