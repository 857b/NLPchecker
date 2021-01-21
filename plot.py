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

def plot_training(losses):
    for name,(period,ys) in losses.items():
        xs = period * np.array(range(len(ys)))
        plt.plot(xs, ys, label=name)
    plt.legend()
    plt.show()
