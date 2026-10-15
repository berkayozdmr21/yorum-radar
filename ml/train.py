"""python -m ml.train --csv datasets/train.csv --test-csv datasets/holdout.csv"""
import argparse
import json
from .model import train

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Train and compare local sentiment classifiers')
    parser.add_argument('--csv',default='datasets/train.csv')
    parser.add_argument('--test-csv')
    parser.add_argument('--output',default='artifacts')
    args=parser.parse_args()
    report=train(args.csv,args.test_csv,args.output)
    print(json.dumps({name:{'macro_f1':result['macro_f1'],'accuracy':result['accuracy']} for name,result in report['metrics'].items()},indent=2))
