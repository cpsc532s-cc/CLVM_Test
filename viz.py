import matplotlib.pyplot as plt
from scipy.misc import imsave
import os, errno
import subprocess

def mkdir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def show_img(img):
    plt.imshow(img)
    plt.show()

class DisplayStream():
    def __init__(self):
        plt.ion()

    def show_img(self, img):
        plt.cla()
        plt.imshow(img)
        plt.pause(0.001)

    def pause(self, dt):
        plt.pause(dt)

class ImageSaver():
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.img_dir_path = os.path.join(self.dir_path, "imgs/")
        mkdir(self.dir_path)
        mkdir(self.img_dir_path)
        self.count = 0

    def save_img(self, img):
        path = os.path.join(self.img_dir_path, "img{}.png".format(str(self.count).zfill(3)))
        imsave(path, img)
        self.count+=1

    def make_gif(self, delay):
        subprocess.call([ 'convert', '-loop', '0', '-delay', str(delay), os.path.join(self.img_dir_path, "img*.png"), os.path.join(self.dir_path, "output.gif")])

