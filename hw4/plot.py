import argparse,pickle
from utlis import *
from PIL import Image
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('--dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--name', type=str)

    args = parser.parse_args()
    return args

args = _parse_args()

with open(os.path.join(args.dir,'log.pkl'), 'rb') as f:
    hist = pickle.load(f)
print(len(hist["D_loss"]))
print(os.path.join("./pic",args.name))
plot_loss(hist,"./pic",args.name)
#
