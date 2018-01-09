# -*- coding: utf-8 -*-
import os
import re
import matplotlib.pyplot as plt
import sys
losses = []
progress = []
style = ['r-', 'b.', 'k--', 'go']
lbl = ['line 1', 'line 2', 'line 3', 'line 4']


def set_ch():  # 设置中文，但仍然失败
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def draw(style='r-'):  # 画图函数
    set_ch()
    plt.figure(figsize=(9, 6))
    plt.plot(progress, losses, style, label='loss with progess')
    # labeldistance，文本的位置离远点有多远，1.1指1.1倍半径的位置
    # autopct，圆里面的文本格式，%3.1f%%表示小数有三位，整数有一位的浮点数
    # shadow，饼是否有阴影
    # startangle，起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看
    # pctdistance，百分比的text离圆心的距离
    plt.title(u'loss with progress \n')
    plt.legend(loc='upper right')  # 设置图例位置
    plt.show()


def readAndDraw(name, n):
    plt.switch_backend('agg')
    plt.figure(figsize=(9, 6))
    file = open(name, 'rb')
    line = file.readline()
    i = 0
    while line:
        progress = []
        losses = []
        if 'loss' not in line:
            line = file.readline()
            continue
        words = line.split('\r')
        print i
        for word in words:
            wd = word.split(' ')
            if len(wd) > 2:
                progress.append(float(wd[1].replace('%', '')))
                losses.append(float(wd[10]))
                # if wd[1] == '1.0%': break
        plt.plot(progress, losses, style[i], label=lbl[i])
        i += 1
        if i >= int(n):
            break
        line = file.readline()

    plt.title(u'loss with progress \n')
    plt.legend(loc='upper right')  # 设置图例位置
    plt.savefig('loss.png')
    #plt.show()


if __name__ == '__main__':
    readAndDraw(sys.argv[1], sys.argv[2])
