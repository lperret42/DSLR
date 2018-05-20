#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

from src.utils import parse_arguments, get_data
from src.dslr import get_description, print_description

def main():
    args = parse_arguments()
    data = get_data(args.csvfile)
    description = get_description(data)
    print_description(description)

if __name__ == '__main__':
    main()
