import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()

data = pd.read_csv(args.path)

plt.plot(data.epoch, data.loss, label='loss')
try:
    plt.plot(data.epoch, data.val_loss, label='val_loss')
except:
    print('No val_loss')
plt.legend()
plt.savefig(args.path.replace('training.log', 'loss.png'))
plt.clf()

plt.plot(data.epoch, data.single_accuracy, label='acc')
try:
    plt.plot(data.epoch, data.val_single_accuracy, label='val_acc')
except:
    print('No val_acc')
plt.legend()
plt.savefig(args.path.replace('training.log', 'acc.png'))
plt.clf()
