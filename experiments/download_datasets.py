import mixnet
import argparse

'''
download raw data
'''

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='All', help='dataset name: ex. [BCIC2a/BCIC2b/BNCI2015_001/SMR_BCI/HighGamma/OpenBMI]')
args = parser.parse_args()

if args.dataset == 'All':
    print("============== All datasets are being downloaded ==========")
    mixnet.utils.load_raw('BCIC2a') 
    mixnet.utils.load_raw('BCIC2b') 
    mixnet.utils.load_raw('BNCI2015_001') 
    mixnet.utils.load_raw('SMR_BCI') 
    mixnet.utils.load_raw('HighGamma') 
    mixnet.utils.load_raw('OpenBMI') 
else: 
    print("============== The {} dataset is being downloaded ==========".format(args.dataset))
    mixnet.utils.load_raw(args.dataset)

'''
1. In case of loading all datasets, please run the script as an example: python download_datasets.py --dataset 'All'
2. In case of loading a particular dataset, please run the script as an example: python download_datasets.py --dataset 'BCIC2a'
'''