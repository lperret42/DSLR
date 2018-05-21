#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

from src.utils import parse_arguments
from  src import dslr
import pandas

def main():
    args = parse_arguments()
    df = dslr.read_csv(args.csvfile)
    df.describe()
    df = pandas.read_csv(args.csvfile)
    print(df.describe())

if __name__ == '__main__':
    main()
