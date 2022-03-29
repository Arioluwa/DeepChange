import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_folder', default='', type=str,
                        help='Path to the folder where the results are saved.')
parser.add_argument('--res_dir', default='./results', help='Path to the folder where the results should be stored')
parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
parser.add_argument('--rdm_seed', default=1, type=int, help='Random seed')
parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
parser.add_argument('--display_step', default=50, type=int,
                        help='Interval in batches between display of training metrics')
# parser.add_argument('--preload', dest='preload', action='store_true',
                        # help='If specified, the whole dataset is loaded to RAM at initialization')
parser.set_defaults(preload=False)
config = parser.parse_args()
config = vars(config)
# print(config.items())
for k, v in config.items():
    print(k)
    print(v)