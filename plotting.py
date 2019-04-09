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

def plot_samples(Y, data, dataset='MNIST'):
    for i in range(10):
        plt.subplot(2,5,i+1)
        string = ''
        for y in Y:
            if dataset=='MNIST':
                string += '\n' + str(y.argmax(1)[i].item()) + ": %.3f" % y[i].max().exp().item()
            elif dataset=='CIFAR10':
                string += '\n' + classes_CIFAR10[y.argmax(1)[i].item()] + ": %.3f" % y[i].max().exp().item()
        plt.title(string)
        if dataset=='MNIST':
            plt.imshow(data[i].squeeze(), cmap='gray', interpolation='none')
        elif dataset=='CIFAR10':
            plt.imshow(data[i].transpose(0,2).transpose(0,1), interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()
    print('\n')