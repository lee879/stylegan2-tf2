
import os
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.callbacks import ModelCheckpoint
# from tensorflow.keras import mixed_precision

from _pilutil import toimage
import glob
from dataset import make_anime_dataset
from dcgan import Generator,Discriminator,LSTM_Discriminator
from stlygan import SNet
from d import D



def generate_big_image(image_data):
    # 将前4张图片拼接成一张大图
    rows = 2
    cols = 2
    channels = 3
    image_size = 256
    big_image = np.zeros((rows * image_size, cols * image_size, channels))
    for i in range(rows):
        for j in range(cols):
            big_image[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size, :] = image_data[
                i * cols + j]

    # 转换为0-255的像素值
    big_image = ((big_image + 1) / 2) * 255
    # big_image = big_image * 255.0
    big_image = big_image.astype(np.uint8)

    return np.expand_dims(big_image,axis=0)
def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    toimage(final_image).save(image_path)

def celoss_ones(logits):
    #热编码
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)

def celoss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)

def  d_loss_fn(generator,discriminator,batch_xy):
    # treat real image  as real
    # treat generator image as fake
    fake_image = generator(None)
    d_fake_logits = discriminator(fake_image)
    d_real_logits = discriminator(batch_xy)
    # d_loss_real = celoss_ones(d_real_logits)
    # d_loss_fake = celoss_zeros(d_fake_logits)
    gp = gradient_penalty(discriminator,batch_xy,fake_image)

    loss = (
            -(tf.reduce_mean(d_real_logits) - tf.reduce_mean(d_fake_logits))
            + 10 * gp
            + (0.001 * tf.reduce_mean(d_real_logits ** 2))
    )
    # loss = d_loss_real + d_loss_fake + 10 * gp

    return loss,gp
def g_loss_fn(generator,discriminator):
    fake_img = generator(None)
    d_fake_logits = discriminator(fake_img,True)
    #gp = gradient_penalty(discriminator, batch_xy, fake_img)
    #
    # loss = celoss_ones(d_fake_logits)
    loss  = - tf.reduce_mean(d_fake_logits)
    return loss , fake_img

def gradient_penalty(discriminator,batch_xy,fake_image): # wgan主要的贡献
    t = tf.random.uniform(batch_xy.shape,minval=0,maxval=1)
    #t = tf.random.normal(batch_xy.shape, mean=0., stddev=1.)
    #t = tf.random.uniforml(batch_xy.shape,minval=-1,maxval=1)
    interplate = t * batch_xy + (1 - t) * fake_image
    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplate_logits = discriminator(interplate)
    grads = tape.gradient(d_interplate_logits,interplate)
    #grads[b,h,w,c]
    grads = tf.reshape(grads,[grads.shape[0],-1])
    gp = tf.norm(grads,axis=1)
    gp = tf.reduce_mean((gp - 1) ** 2)
    return gp

#使用余弦退火降低学习率
class CosineAnnealingSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_max, lr_min, T):
        super(CosineAnnealingSchedule, self).__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T = T

    def __call__(self, step):

        t = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos((step/self.T) * np.pi))
        print("step{},lr;{}".format(step,t))
        return t

def main():
    #mixed_precision.set_global_policy('mixed_float16')
    # hyper parameters

    # tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    z_dim = 512
    epochs = 88888888
    batch_size = 12# 更具显存换批次跑()
    is_training = True
    summary_writer = tf.summary.create_file_writer(r".\log")

    #img_path = glob.glob(r"D:\pj\seeprettyface_age_kids\age_kids\*.*")

    # img = dataGenerator("age_kids",im_size=256)
    img_path = glob.glob(r"D:\pj\321\data\age_kids\*.*")
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size,shuffle= batch_size * 8,resize = 256)  # 自己建立的数据划分
    dataset = dataset.repeat()
    db_iter = iter(dataset)

    generator = SNet(batch=batch_size,s_layers=6)
    discriminator = D()

    y = generator(None)
    x = tf.random.normal([16,256,256,3])
    y = discriminator(x)
    # generator.load_weights(r"D:\pj\321\ckpt\best_g.h5")
    # discriminator.load_weights(r"D:\pj\321\ckpt\best_d.h5")

    checkpoint_dir = './ckpt'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    # 将优化器包装在 LossScaleOptimizer 中
    # optimizer_d = tf.keras.mixed_precision.LossScaleOptimizer(optimizer_d, dynamic=True)
    # optimizer_g = tf.keras.mixed_precision.LossScaleOptimizer(optimizer_g, dynamic=True)

    # Define the checkpoint to save the model with the best weights based on validation loss
    best_weights_checkpoint_path_d = os.path.join(checkpoint_dir, 'best_d.h5')
    best_weights_checkpoint_path_g = os.path.join(checkpoint_dir, 'best_g.h5')
    best_weights_checkpoint_g = ModelCheckpoint(best_weights_checkpoint_path_g,
                                              monitor='loss',
                                              verbose=1,
                                              save_best_only=True,
                                              save_weights_only=True,
                                              mode='min')
    best_weights_checkpoint_d = ModelCheckpoint(best_weights_checkpoint_path_d,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=True,
                                                mode='min')

    for epoch in range(epochs):

        #batch_xy = img.get_batch(batch_size)
        batch_xy = next(db_iter)

        with tf.GradientTape() as tape:
            d_loss, gp = d_loss_fn(generator, discriminator, batch_xy)
        d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
        tf.optimizers.Adam(learning_rate=2e-4, beta_1=0.5,beta_2=0.99).apply_gradients(zip(d_grads, discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            g_loss, fake_img = g_loss_fn(generator, discriminator)
        g_grads = tape.gradient(g_loss, generator.trainable_variables)
        tf.optimizers.Adam(learning_rate=2e-4, beta_1=0.5,beta_2=0.99).apply_gradients(zip(g_grads, generator.trainable_variables))


        with summary_writer.as_default():
            tf.summary.scalar('d_loss', float(d_loss), step=epoch)
            tf.summary.scalar('g_loss', float(g_loss), step=epoch)
            print(epoch, "d_loss:", d_loss, "g_loss", float(g_loss), "gd", float(gp))
            generator.save_weights(best_weights_checkpoint_path_g)
            discriminator.save_weights(best_weights_checkpoint_path_d)

            if epoch % 100 == 0:
                #print("To add imag in tensorboard .......")
                img1 = generate_big_image(fake_img)
                tf.summary.image("fake_image", img1, step=epoch)
                img2 = generate_big_image(batch_xy)
                tf.summary.image("real_image", img2, step=epoch)
                tf.keras.backend.clear_session()
                print("tf.keras.backend.clear_session")

def predict():
    generator = Generator()
    generator.build(input_shape=(100, 100))
    generator.load_weights(r"")
    z = tf.random.normal([100,100],mean=0,stddev=1)
    fake_iamge = generator(z, training=False)
    img_path = os.path.join(r"./predict", "testv4.png" )
    save_result(fake_iamge.numpy(), 5, img_path, color_mode="P")

if __name__ == "__main__":
    main()
    #predict()
    # generator.load_weights(best_model_checkpoint_path)