# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:22:18 2017

@author: Cy
本程序来自
http://blog.topspeedsnail.com/archives/10858
TensorFlow练习20: 使用深度学习破解字符验证码
"""
import random
import os
from train_model import crack_captcha_cnn, MAX_CAPTCHA, IMAGE_HEIGHT, IMAGE_WIDTH, CHAR_SET_LEN, X, keep_prob, vec2text, convert2gray
from generate_captcha import sample_captcha
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import time
#import crack_captcha
from PIL import Image


def crack_captcha1(captcha_image):
    #from tensorflow.python.framework import ops
    # ops.reset_default_graph()
    output = crack_captcha_cnn()
    checkpoint_dir = "D:/tmp/tf_crack_cjt_captcha_model/crack_captcha_0.9725.model-9300"
    # tf.reset_default_graph()
    saver = tf.train.Saver()
    #g = tf.Graph()
    #sess = tf.InteractiveSession(graph=g)
    # with tf.Session() as sess:

    sess = tf.InteractiveSession()
    #saver.restore(sess, tf.train.latest_checkpoint('.'))
    saver.restore(sess, checkpoint_dir)
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

    text = text_list[0].tolist()
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1

    return vec2text(vector)


def gen_captcha_test(size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    text, image = sample_captcha(size=size)
    image = convert2gray(image)
    image = image.flatten() / 255
    predict_text = crack_captcha1(image)
    print('正确验证码:%s, 识别验证码为:%s' % (text, predict_text))
    if text == predict_text:
        print('正确识别')
    else:
        print('错误')


def cjt_img_test(cjt_img_path=r'D:\Program Files\Tesseract-OCR\cjtdata\4.gif'):
    cjt_img = Image.open(cjt_img_path)
    plt.subplot(1, 3, 1)
    plt.imshow(cjt_img)
    cjt_img_resized = cjt_img.resize([IMAGE_WIDTH, IMAGE_HEIGHT])
    plt.subplot(1, 3, 2)
    plt.imshow(cjt_img_resized)
    cjt_img_arr = np.asarray(cjt_img_resized)
    cjt_img_cvt2gray = convert2gray(cjt_img_arr)
    plt.subplot(1, 3, 3)
    plt.imshow(cjt_img_cvt2gray)
    plt.show()
    image = cjt_img_cvt2gray.flatten() / 255
    predict_text = crack_captcha1(image)
    print('识别验证码为:%s' % predict_text)
    return cjt_img, predict_text


def cjt_img_random_test(cjt_img_dirs=r'D:\\Program Files\\Tesseract-OCR\\cjtdata\\'):
    rnd_num = random.randint(1000, 9999)
    cjt_img_path = '%s%s.gif' % (cjt_img_dirs, rnd_num)
    if os.path.exists(cjt_img_path):
        cjt_img_test(cjt_img_path)
    else:
        print('无此验证码照片')


if __name__ == '__main__':
    #pass
    gen_captcha_test()

#==============================================================================
# _time = time.time()
# #for i in range(100):
# #tf.reset_default_graph()
# text, image = gen_captcha_text_and_image()
# imgshow = image
# image = convert2gray(image)
# image = image.flatten() / 255
# predict_text = crack_captcha.crack_captcha(image)
#
# print("正确: {}  预测: {},".format(text, predict_text))
#==============================================================================
#i +=1
#print("共耗时: {}".format( time.time()-_time))
#f = plt.figure()
#ax = f.add_subplot(111)
#ax.text(0.1, 0.9,text, ha='center', va='center', transform=ax.transAxes)
# plt.imshow(imgshow)
# plt.show()
