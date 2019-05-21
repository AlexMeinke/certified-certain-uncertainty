import matplotlib.pyplot as plt
from matplotlib import patches

#plot_2d_sample(X_MNIST,model.gmm.mu, (model.gmm.logvar/2).exp()/100,idx=(345,346))
def plot_2d_sample(sample, mu=None, sig=None, idx=(0,1)):
    """
    :param sample: 
    :param mu:
    :param sig: 
    :return:
    """
    ax = plt.gca()
    ax.cla()
    
    sample_np = sample.numpy()
    x = sample_np[:, idx[0]]
    y = sample_np[:, idx[1]]
    ax.scatter(x, y)

    if (mu is not None and sig is not None):
        for i in range(mu.shape[0]):
            ellipse = patches.Ellipse( (mu[i,idx[0]], mu[i,idx[1]]) , 2*sig[i], 2*sig[i], color='r', fill=False) 
            ax.add_artist(ellipse)
    plt.axis('equal')
    plt.show()


def plot_samples(Y, data, dataset='MNIST'):
    for i in range(10):
        plt.subplot(2,5,i+1)
        string = ''
        for y in Y:
            string += ('\n' 
                       + classes_dict[dataset][y.argmax(1)[i].item()] 
                       + ": %.3f" % y[i].max().exp().item() )

        plt.title(string)
        if dataset in ['MNIST', 'FMNIST']:
            plt.imshow(data[i].squeeze().detach().cpu(), cmap='gray', interpolation='none')
        elif dataset in ['CIFAR10', 'SVHN', 'CIFAR100']:
            plt.imshow(data[i].transpose(0,2).transpose(0,1).detach().cpu(), interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()
    print('\n')
    
classes_FMNIST = (
'shirt',
'trousers',
'pullover',
'dress',
'coat',
'sandal',
'shirt',
'sneaker',
'bag',
'boot')
    
classes_CIFAR10 = ('plane',
 'car',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck')

classes_CIFAR100 = ['apple',
 'aquarium_fish',
 'baby',
 'bear',
 'beaver',
 'bed',
 'bee',
 'beetle',
 'bicycle',
 'bottle',
 'bowl',
 'boy',
 'bridge',
 'bus',
 'butterfly',
 'camel',
 'can',
 'castle',
 'caterpillar',
 'cattle',
 'chair',
 'chimpanzee',
 'clock',
 'cloud',
 'cockroach',
 'couch',
 'crab',
 'crocodile',
 'cup',
 'dinosaur',
 'dolphin',
 'elephant',
 'flatfish',
 'forest',
 'fox',
 'girl',
 'hamster',
 'house',
 'kangaroo',
 'keyboard',
 'lamp',
 'lawn_mower',
 'leopard',
 'lion',
 'lizard',
 'lobster',
 'man',
 'maple_tree',
 'motorcycle',
 'mountain',
 'mouse',
 'mushroom',
 'oak_tree',
 'orange',
 'orchid',
 'otter',
 'palm_tree',
 'pear',
 'pickup_truck',
 'pine_tree',
 'plain',
 'plate',
 'poppy',
 'porcupine',
 'possum',
 'rabbit',
 'raccoon',
 'ray',
 'road',
 'rocket',
 'rose',
 'sea',
 'seal',
 'shark',
 'shrew',
 'skunk',
 'skyscraper',
 'snail',
 'snake',
 'spider',
 'squirrel',
 'streetcar',
 'sunflower',
 'sweet_pepper',
 'table',
 'tank',
 'telephone',
 'television',
 'tiger',
 'tractor',
 'train',
 'trout',
 'tulip',
 'turtle',
 'wardrobe',
 'whale',
 'willow_tree',
 'wolf',
 'woman',
 'worm']


classes_dict = {'MNIST':          list(range(10)),
               'FMNIST':         classes_FMNIST,
               'SVHN':           list(range(10)),
               'CIFAR10':        classes_CIFAR10,
               'CIFAR100':       classes_CIFAR100,
              }
