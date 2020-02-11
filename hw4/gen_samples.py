# from ACGAN3 import *
# from ACGAN31 import *
# from ACGAN5 import * 0103 pass base line
# from ACGAN5_2 import *
from ACGAN5_3 import * #/2340
# from ACGAN6 import *
from cartoonDataset import cartoonDataset

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('--dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--output_dir', type=str,
                        )
    parser.add_argument('--label', type=str,
                        )

    parser.add_argument('--epoch', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    epoch = args.epoch
    cartoon_data = cartoonDataset()
    gan = ACGAN(200,64,cartoon_data,label_file=args.label)
    gan.load(epoch=epoch,path="")
    gan.generate_samples(save_dir=args.output_dir,
                            epoch=epoch,label_file = args.label)
