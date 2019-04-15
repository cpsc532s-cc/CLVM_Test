from torchvision import datasets, transforms
import viz

def load_mnist():
    mnist = datasets.MNIST('./data', train=True, download=True,
        transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
        ]))
    return mnist.train_data

if __name__ == "__main__":
    mnist = load_mnist()
    print(mnist.shape)
    #viz.show_img(mnist[0])
    """
    ds = viz.DisplayStream()
    for i in range(1000):
        ds.show_img(mnist[i])
        ds.pause(0.1)
    """

    ds = viz.ImageSaver("./figs/run1")
    for i in range(100):
        ds.save_img(mnist[i])
    ds.make_gif()
