import argparse
import models
import utils

parser = argparse.ArgumentParser(description='Test the synthetic PPG model')
parser.add_argument('--dataset', type=str, default='BIDMC_test', help='Dataset: [BIDMC_test, CAPNO_test, DALIA_test, WESAD_test, PTT_test]')
parser.add_argument('--window_size', type=int, default=4, help='Window size: [4, 8, 16, 32, 64]')
args = parser.parse_args()

assert args.dataset in ['BIDMC_test', 'CAPNO_test', 'DALIA_test', 'WESAD_test', 'PTT_test'], 'Unknown dataset'
assert args.window_size in [4, 8, 16, 32, 64], 'Unknown window size'

if args.dataset == 'PTT_test':
    for split in ['all', 'sit', 'walk', 'run']:
        test_ppg, test_ecg =  utils.train_test_split(dataset=args.dataset, split=split, window_size=args.window_size)
        representative_ppg, heart_rates = models.evaluate(test_ppg)
        gt_hr, pred_hr = models.get_HR(test_ecg, representative_ppg, args.window_size)
        mae_hr = models.MAE_HR(test_ecg,representative_ppg,args.window_size)
else:
    test_ppg, test_ecg =  utils.train_test_split(dataset=args.dataset, window_size=args.window_size)
    representative_ppg, heart_rates = models.evaluate(test_ppg)
    gt_hr, pred_hr = models.get_HR(test_ecg, representative_ppg, args.window_size)
    mae_hr = models.MAE_HR(test_ecg,representative_ppg,args.window_size)