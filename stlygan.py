import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer,Dense,Conv2D,Conv2DTranspose,UpSampling2D,AveragePooling2D,Activation,LeakyReLU
from tensorflow.keras import Model,Sequential,layers
from conv_mod import Conv2DMod
"""
本次设计未采用渐进式训练，而是采用了skip
"""

class EqualLinear(Layer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 bias_init=0,
                 lr_mul=1,
                 activation=True):
        super(EqualLinear, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.bias_init = bias_init
        self.lr_mul = lr_mul
        self.activation = activation

    def build(self, input_shape):
        self.weight = self.add_weight(name="w",shape=(self.in_dim, self.out_dim),
                                      initializer=tf.initializers.RandomNormal(),
                                      trainable=True)
        self.weight.assign(self.weight / self.lr_mul)

        if self.bias:
            self.bias = self.add_weight("b",shape=(self.out_dim,),
                                        initializer=tf.initializers.Constant(self.bias_init),
                                        trainable=True)
        else:
            self.bias = None

        self.scale = (1 / math.sqrt(self.in_dim)) * self.lr_mul

        super(EqualLinear, self).build(input_shape)

    def call(self, inputs):
        if self.activation:
            inputs = tf.cast(inputs,tf.float32)
            out = tf.matmul(inputs, (self.weight * self.scale))
            out = self.fused_leaky_relu(out, bias=self.bias * self.lr_mul)
        else:
            out = tf.matmul(inputs, (self.weight * self.scale))
            if self.bias is not None:
                out += self.bias * self.lr_mul
        return out

    def fused_leaky_relu(self, input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
        if bias is not None:
            rest_dim = tf.ones(len(input.shape) - len(bias.shape) - 1, dtype=tf.int32)
            return (tf.nn.leaky_relu(input + tf.reshape(bias, (1, bias.shape[0], *rest_dim)),
                                     alpha=negative_slope) * scale)
        else:
            return tf.nn.leaky_relu(input, alpha=negative_slope) * scale


class PixelNorm(Layer):  #这一层主要是进行标准化, tf自带的bn有很大的问题，
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs * tf.math.rsqrt(
            tf.math.mean(inputs * inputs, 1, keepdim=True) + 1e-8)

class Laten(Layer): # 这里就是一个常量数据 在CGAN中提出
    def __init__(self, channel, size=4):
        super(Laten, self).__init__()

        self.inp = self.add_weight(shape=(1, size, size,channel),
                                     initializer=tf.initializers.RandomNormal(),
                                     trainable=False)

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        out = tf.tile(self.inp, multiples=(batch, 1, 1, 1))

        return out

class StyleConv2d(Layer):
    def __init__(self,indx = 1,xz = 1):
        super(StyleConv2d, self).__init__()
        self.channels = [512,512,512,512,256,128,64,32,16,16]
        if xz == 1:
            self.equalLinear = EqualLinear(in_dim=512,out_dim=self.channels[indx],bias=1) #主要解决style shape的问题
        else:
            self.equalLinear = EqualLinear(in_dim=512, out_dim=self.channels[indx+1], bias=1)  # 主要解决style shape的问题

        self.convstyle = Conv2DMod(filters=self.channels[indx + 1],kernel_size=3)
        self.pn = PixelNorm()
    def call(self, inputs,style, *args, **kwargs):
        style = self.pn(self.equalLinear(style)) # 对单个style仿射变换
        out = self.convstyle([inputs,style]) # style 的融合
        return out

class Style(Layer):
    def __init__(self,batch,s_layers = 16):
        super(Style, self).__init__()
        self.noise = Noise(batch=batch)
        self.mapping = Mapping()
        self.s_layers = s_layers
        self.laten = Laten(channel=512)

    def call(self, inputs = None, *args, **kwargs):
        styles_noise = self.noise(None) # 返回的两个列表
        styles = [self.mapping(s_n) for s_n in styles_noise] # 得到两个 [batch,512] 的styles
        indx = np.random.randint(1,self.s_layers + 1)

        styles_laten1 = tf.expand_dims(styles[0],axis=1)
        styles_laten1 = tf.tile(styles_laten1,[1,indx,1])

        styles_laten2 = tf.expand_dims(styles[1],axis=1)
        styles_laten2 = tf.tile(styles_laten2, [1, self.s_layers-indx, 1])

        style_laten = tf.concat([styles_laten1,styles_laten2],axis=1)

        laten = self.laten(style_laten) # laten 无所谓是个标量就行了，无需参与训练

        return laten,style_laten

class To_RGB(Layer):
    def __init__(self,indx = 0):
        super(To_RGB, self).__init__()
        self.f = [512,512,512,256,128,64,32,16]
        self.equalLinears = EqualLinear(in_dim=512,out_dim=self.f[indx])
        self.conv = Conv2DMod(3,1)
    def call(self, inputs, style,*args, **kwargs):
        style = self.equalLinears(style)
        out = self.conv([inputs,style])
        return out

class up(Layer):
    def __init__(self):
        super(up, self).__init__()
        self.up = UpSampling2D(2)

    def call(self, inputs, *args, **kwargs):
        out = self.up(inputs)
        return out

class down(Layer):
    def __init__(self):
        super(down, self).__init__()
        self.down = AveragePooling2D(2)
    def call(self, inputs, *args, **kwargs):
        out = self.down(inputs)
        return out

class LowPassFilterLayer(tf.keras.layers.Layer):
    '''
    低通滤波器
    '''
    def __init__(self, **kwargs):
        super(LowPassFilterLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        _, height, width, _ = input_shape

        # 计算图像中心点坐标
        center_row, center_col = height // 2, width // 2

        # 创建截止频率参数
        self.cutoff_frequency = 16 # 这个

        # 创建掩膜
        mask = np.zeros((height, width), np.float32)
        cutoff_frequency_value = tf.cast(self.cutoff_frequency, tf.int32).numpy()
        mask[center_row - cutoff_frequency_value: center_row + cutoff_frequency_value,
             center_col - cutoff_frequency_value: center_col + cutoff_frequency_value] = 1

        # 扩展掩膜到匹配图像的通道数
        mask = np.expand_dims(mask, axis=-1)
        mask = np.tile(mask, (1, 1, input_shape[-1]))

        # 创建掩膜的Tensor并转换为复数类型
        self.mask = tf.constant(mask, dtype=tf.complex64)

    def call(self, inputs):
        # 进行傅里叶变换
        fft = tf.signal.fft2d(tf.cast(inputs, tf.complex64))
        fft_shifted = tf.signal.fftshift(fft)

        # 将掩膜转换为复数类型
        mask_complex = tf.cast(self.mask, tf.complex64)

        # 应用掩膜
        filtered_shifted = fft_shifted * mask_complex

        # 进行逆向傅里叶变换
        filtered = tf.signal.ifftshift(filtered_shifted)
        filtered_image_complex = tf.signal.ifft2d(filtered)

        # 提取实部并取绝对值得到滤波后的图像张量
        filtered_image = tf.math.abs(tf.cast(filtered_image_complex, tf.float32))

        return filtered_image


class NoiseNet(Layer):
    def __init__(self):
        super(NoiseNet, self).__init__()

    def build(self, input_shape):
        self.scal = self.add_weight(name='scal', shape=(),
                                    initializer=tf.keras.initializers.Constant(value=0.0),
                                    trainable=True)
        super(NoiseNet, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        batch, height, width, _ = inputs.shape
        noise = tf.random.normal([batch, height, width, 1])
        return noise * self.scal + inputs


class Mapping(Layer): # 这里将产生（batch,512）个Mapping
    def __init__(self,layers=8):
        super(Mapping, self).__init__()
        self.pn = PixelNorm()
        self.equalLinears = [EqualLinear(in_dim=512,out_dim=512) for _ in range(layers)]
        self.layers = layers

    def call(self, inputs, training=None, mask=None):
        x = self.pn(inputs)
        for i in range(self.layers):
            x = self.equalLinears[i](x)
        out = self.pn(x)
        return out

class Noise(Layer): # 返回的是两个噪声列表2个[batch,512] 的噪声
    def __init__(self,batch):
        super(Noise, self).__init__()
        self.num_style_feat = 512
        self.batch = batch
        self.num_noise = 2

    def call(self, inputs=None, training=None, mask=None):
        out = self.make_noise(self.batch,2)
        return out

    def make_noise(self, batch, num_noise):
        if num_noise == 1:
            noises = tf.random.normal([batch, self.num_style_feat])
        else:
            noises = []
            for _ in range(num_noise):
                noises.append(tf.random.normal([batch, self.num_style_feat]))
        return noises

class SNet(Model):
    def __init__(self,batch,s_layers= 6): # 这里控制输出的图片的大小
        super(SNet, self).__init__()
        self.style = Style(batch, s_layers=2 * (s_layers+1) + 1) # 这里控制style的层数

        self.init_styleconv = StyleConv2d(indx=0)
        self.init_to_rgb = To_RGB(indx=0)

        self.styleconv1 = [StyleConv2d(indx = i,xz=1) for i in range(s_layers)]
        self.ac = LeakyReLU(0.2)
        self.styleconv2 = [StyleConv2d(indx = i,xz=2) for i in range(s_layers)]
        self.to_rgb = [To_RGB(indx=i) for i in range(s_layers)]
        self.noise1 = [NoiseNet() for _ in range(s_layers)]
        self.noise2 = [NoiseNet() for _ in range(s_layers)]
        self.pn = PixelNorm()
        #self.bn = layers.BatchNormalization()

        self.up = up()
        #self.bur = [LowPassFilterLayer() for _ in range(s_layers)]

        self.s_layers = s_layers


    def call(self, inputs, training=None, mask=None):
        laten , style = self.style(None)
        IMG = []
        x = self.init_styleconv(laten,style[:,0,:])
        img = self.init_to_rgb(x,style[:,1,:])
        IMG.append(img)
        temp_key = 2
        for i in range(self.s_layers):
            x = self.up(x)
            #x = self.bur[i](x)

            x = self.styleconv1[i](x,style[:,temp_key,:])
            x = self.noise1[i](x)
            x = self.ac(x)
            #x = self.pn(x)

            x = self.styleconv2[i](x,style[:,temp_key+1,:])
            x = self.noise2[i](x)
            x = self.ac(x)
            #x = self.pn(x)

            img = tf.tanh(self.to_rgb[i](x,style[:,temp_key+2,:]))
            IMG.append(img)
            temp_key += 2

        img = IMG[0] # 将第一个照片给搞出来
        for i in range(len(IMG)-1):
            x = self.up(img)
            #x = self.bur[i](x)
            img = layers.add([x,IMG[i+1]])

        return tf.tanh(img) #(img / 2) + 0.5 # tf.tanh(img)

# model = SNet(batch=8,s_layers=7)
# y = model(None)
# y = y.numpy()
# # model.summary()
# print(y.shape)
# # 获取每一层的名称
# for i, w in enumerate(model.weights):
#     print(i, w.name)