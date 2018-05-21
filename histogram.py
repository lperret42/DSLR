#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

from src.utils import parse_arguments, get_data, is_float
from src import dslr
import matplotlib.pyplot as plt

def main():
    args = parse_arguments()
    df = dslr.read_csv(args.csvfile)
    df.remove_nan()
    df.standardize()
    df_standardized = dslr.DataFrame(data=df.standardized)
    dfs_by_house = {house: df_standardized.get_df_filtered({"Hogwarts House": house}) for
            house in list(set(df.data["Hogwarts House"]))}
    for feature, values in df.data.items():
        if feature != "Index" and is_float(values[0]):
            to_plot = [df.data[feature] for house, df in dfs_by_house.items()]
            plt.hist(to_plot, color=['b', 'g', 'r', 'c'][:len(to_plot)])
            plt.xlabel("Notes")
            plt.ylabel("Frequency")
            plt.legend([house for house, _ in dfs_by_house.items()])
            plt.title(feature)
            plt.show()

if __name__ == '__main__':
    main()
