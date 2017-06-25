import sys
import simple_test
import torch
import argparse

def main(argv):
    use_gpu = False 
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print('CUDA not detected using CPU') 
        use_gpu = False 
    simple_test.test_gaussian(1,2,100000, make_gif=True, use_gpu=use_gpu)

if __name__ == '__main__':
    main(sys.argv)
