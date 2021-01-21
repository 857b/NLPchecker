import matplotlib.pyplot as plt
import numpy as np

def plot_maskedLogitError(outputs):
    def cumulated(vs):
        xs = np.sort(vs)
        ys = np.array(range(1,len(vs)+1)) / len(vs)
        return xs,ys
    ccx,ccy = cumulated(outputs[0])
    ecx,ecy = cumulated(outputs[1])
    ccy = 100 * ccy       # false positive
    ecy = 100 * (1 - ecy) # false negative
    plt.plot(ccx, ccy, label='false positive')
    plt.plot(ecx, ecy, label='false negative')
    plt.xlabel('threshold')
    plt.ylabel('%')
    plt.legend()
    plt.show()

def plot_training(losses, num_train):
    ex = num_train * (0.5 + np.array(range(len(losses['epoch_loss']))))
    plt.plot(ex, losses['epoch_loss'], label='train epoch loss')

    px = np.cumsum(losses['periods'])
    plt.plot(px, losses['period_loss'], label='train period loss')

    if losses['test_loss']:
        plt.plot(px, losses['test_loss'], label='test loss')
    if losses['test_acc']:
        plt.plot(px, losses['test_acc'], label='test acc')

    plt.xlabel('sample')
    
    plt.legend()
    plt.show()

def plot_training2(l1, l2, num_train, shift2):
    ex = num_train * (0.5
            + np.array(range(len(l1['epoch_loss']) + len(l2['epoch_loss']))))
    plt.plot(ex, l1['epoch_loss'] + l2['epoch_loss'], label='train epoch loss')

    px = np.concatenate([np.cumsum(l1['periods']),
        shift2 + np.cumsum(l2['periods'])])
    plt.plot(px, l1['period_loss']+l2['period_loss'], label='train period loss')

    plt.plot(px, l1['test_loss']+l2['test_loss'], label='test loss')
    plt.plot(px, l1['test_acc']+l2['test_acc'], label='test acc')

    plt.xlabel('sample')
    
    plt.legend()
    plt.show()
