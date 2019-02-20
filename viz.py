import matplotlib.pyplot as plt

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
