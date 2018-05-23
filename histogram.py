#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

import argparse
import matplotlib.pyplot as plt
from src import dslr

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', help='data.csv')
    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    df = dslr.read_csv(args.csvfile)
    df.get_numerical_features()
    df.remove_nan()
    df.standardize()
    df_standardized = dslr.DataFrame(data=df.standardized)
    dfs_by_house = {house: df_standardized.get_df_filtered({"Hogwarts House": house}) for
            house in list(set(df.data["Hogwarts House"]))}
    for feature in df.numerical_features:
        values = df.data[feature]
        to_plot = [df.data[feature] for house, df in dfs_by_house.items()]
        plt.hist(to_plot)
        plt.xlabel("Notes")
        plt.ylabel("Frequency")
        plt.legend([house for house, _ in dfs_by_house.items()])
        plt.title(feature)
        plt.show()

if __name__ == '__main__':
    main()
