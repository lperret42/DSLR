import argparse
from src import dslr

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', help='data.csv')
    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    df = dslr.read_csv(args.csvfile)
    df.describe()

if __name__ == '__main__':
    main()
