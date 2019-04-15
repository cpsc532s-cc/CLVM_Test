import matplotlib
#matplotlib.use('TkAgg')
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
    plt.imshow(img,cmap="grey")
    plt.show()

def mypause(interval):
    manager = plt._pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()        
        canvas.start_event_loop(interval)
    else:
        time.sleep(interval)

class DisplayStream():
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        self.first_time = True
        self.ax = self.fig.add_subplot(1,1,1) 

    def show_img(self, img):
        if self.first_time:
            self.first_time = False
            self.fig.show()

        self.ax.cla()
        self.ax.imshow(img,cmap="gray")
        #self.pause(0.0001)
        self.fig.canvas.flush_events()

    def pause(self, dt):
        mypause(dt)

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

