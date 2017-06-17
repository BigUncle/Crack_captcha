# coding:utf-8
import os
import math
import random
import shutil
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# abcdefghjkmnpqrstuvwxy
_letter_cases = "abdefghmnpqrstwxyz"  # 小写字母，去除可能干扰的c i j k l o u v
_upper_cases = "ABDEFHMNPQRSTWXYZ"  # 大写字母，去除可能干扰的C G I J K L O U V
_numbers = ''.join(map(str, range(2, 10)))  # 数字，去除0，1
init_chars = ''.join((_letter_cases, _upper_cases, _numbers))
current_dir = os.path.dirname(__file__)
fontType = os.path.join(current_dir, "luxirb.ttf")
bg_image = os.path.join(current_dir, "background.jpg")
out_dir = os.path.join(current_dir, "mycaptchas")
import matplotlib.pyplot as plt

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def create_validate_code(size=(140, 65),
                         chars=init_chars,
                         bg_image=bg_image,
                         bg_color=tuple(np.random.randint(180, 255, size=3)),
                         #fg_color=tuple(np.random.randint(40, 179, size=3)),
                         fg_color=(255, 255, 255),
                         font_size=19,
                         font_type=fontType,
                         font_color=(0, 0, 0), #tuple(np.random.randint(50, 180, size=3)),
                         char_length=4,
                         draw_lines=True,
                         n_line=(10, 16),
                         min_length=1,
                         max_length=12,
                         draw_points=False,
                         point_chance=2):
    '''
    @todo: 生成验证码图片
    @param size: 图片的大小，格式（宽，高），默认为(120, 30)
    @param chars: 允许的字符集合，格式字符串
    @param img_type: 图片保存的格式，默认为GIF，可选的为GIF，JPEG，TIFF，PNG
    @param mode: 图片模式，默认为RGB
    @param bg_color: 背景颜色，默认为白色
    @param fg_color: 前景色，验证码字符颜色，默认为白色#FFFFFF
    @param font_size: 验证码字体大小
    @param font_type: 验证码字体，默认为 ae_AlArabiya.ttf
    @param length: 验证码字符个数
    @param draw_lines: 是否划干扰线
    @param n_lines: 干扰线的条数范围，格式元组，默认为(1, 2)，只有draw_lines为True时有效
    @param min_length: 干扰线的最小长度
    @param max_length: 干扰线的最大长度
    @param draw_points: 是否画干扰点
    @param point_chance: 干扰点出现的概率，大小范围[0, 100]
    @return: [0]: PIL Image实例
    @return: [1]: 验证码图片中的字符串
    '''

    width, height = (int(x*0.9) for x in size)  # 宽， 高
    #print(width, height)
    #img = Image.open(bg_image)  # 创建图形
    img = Image.new(mode='RGB', size=size, color=bg_color)
    draw = ImageDraw.Draw(img)  # 创建画笔

    strs = create_strs(draw, chars, char_length, font_type, font_size, width, height, fg_color)
    # # 图形扭曲参数
    if draw_points:
        create_points(draw, point_chance, width, height)
    if draw_lines:
        create_lines(draw, min_length, max_length, n_line, width, height)

    params = [1 - float(random.randint(1, 2)) / 100,
               0,
               0,
               0,
               1 - float(random.randint(1, 10)) / 100,
               float(random.randint(1, 2)) / 500,
               0.001,
               float(random.randint(1, 2)) / 500
               ]
    img = img.transform(size, Image.PERSPECTIVE, params) # 创建扭曲
    #img = img.filter(ImageFilter.DETAIL)  # 滤镜，边界加强（阈值更大）
    return img, strs


def create_lines(draw, min_length, max_length, n_line, width, height):
    '''绘制干扰线'''
    main_line_color=tuple(np.random.randint(1, 179, size=3))
    line_num = random.randint(n_line[0], n_line[1])  # 干扰线条数
    main_begin = (0, random.randint(0, height))
    main_end = (width, random.randint(0, height))
    #print('endding point:', end)
    draw.line([main_begin, main_end], fill=main_line_color, width=4)

    sub_line_color=tuple(np.random.randint(150, 255, size=3))
    for i in range(line_num):
        # 起始点
        begin = (random.randint(0, width), random.randint(0, height))
        # 长度
        length = min_length + random.random() * (max_length - min_length)
        # 角度
        alpha = random.randrange(0, 180)
        # 结束点
        end = (begin[0] + length * math.cos(math.radians(alpha)),
               begin[1] - length * math.sin(math.radians(alpha)))
        #end = (width, random.randint(0, height))
        draw.line([begin, end], fill=sub_line_color, width=1)


def create_points(draw, point_chance, width, height):
    '''绘制干扰点'''
    chance = min(100, max(0, int(point_chance)))  # 大小限制在[0, 100]

    for w in range(width):
        for h in range(height):
            tmp = random.randint(0, 100)
            if tmp > 100 - chance:
                draw.point((w, h), fill=(0, 0, 0))


def create_strs(draw, chars, char_length, font_type, font_size, width, height, fg_color):
    '''绘制验证码字符'''
    '''生成给定长度的字符串，返回列表格式'''
    # c_chars = random.sample(chars, length) # sample产生的是unique的char
    flag = False
    while not flag:
        c_chars = np.random.choice(list(chars), char_length).tolist()
        strs = ''.join(c_chars)  # 每个字符前后以空格隔开

        font = ImageFont.truetype(font_type, font_size)
        font_width, font_height = font.getsize(strs)

        try:
#==============================================================================
#             start_x = random.randrange(0, width - font_width)
#             start_y = random.randrange(0, height - font_height)
#==============================================================================
            start_x = random.randrange(10, 20)
            start_y = random.randrange(0, 2)
        except ValueError as e:
            print(e)
            print(strs)
            print(width, font_width, height, font_height)
        else:
            flag = True

    draw.text((start_x, start_y), strs, font=font, fill=fg_color, spacing=10)
    return ''.join(c_chars)


def binarization(image):
    binarized_img = Image.new("L", size=image.size)
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            r, g, b = image.convert('RGB').getpixel((i, j))
            value = int(0.299 * r + 0.587 * g + 0.114 * b)
            if value < 180:
                binarized_img.putpixel((i, j), 255)
            else:
                binarized_img.putpixel((i, j), 0)
    return binarized_img


if __name__ == "__main__":
    #test_image = Image.open("/home/lan/Desktop/test1.jpg")  # /home/lan/PycharmProjects/cnn-for-captcha/CaptchaGenerator/mycaptchas/5WMn6m.jpg
    #binarization(test_image).save("/home/lan/Desktop/test_bi.jpg")
    cpt_cnt = 24
    plt.figure(num='astronaut',figsize=(28,10))
    plt.axis('off')
    for i in range(cpt_cnt):
        x, y =create_validate_code(size=(146, 65), font_size=40, bg_color=tuple(np.random.randint(180, 255, size=3)), fg_color=tuple(np.random.randint(40, 179, size=3)),font_type='c:\\windows\\fonts\\ARIALN.ttf', char_length=4, draw_points=True, point_chance=4, draw_lines=True, n_line=(8, 12), min_length=15, max_length=30)
        plt.subplot(6, cpt_cnt//6, i+1)
        plt.imshow(x)
    plt.show()