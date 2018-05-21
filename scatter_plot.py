#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

from src.utils import parse_arguments, get_data, is_float
from src import dslr
import matplotlib.pyplot as plt

def main():
    args = parse_arguments()
    df = dslr.read_csv(args.csvfile)
    df.remove_nan()
    df.standardize()
    numerical_features = [feature for feature, values in df.data.items() if
                    feature != "Index" and is_float(values[0])]
    df_standardized = dslr.DataFrame(data=df.standardized)
    dfs_by_house = {house: df_standardized.get_df_filtered({"Hogwarts House": house}) for
            house in list(set(df.data["Hogwarts House"]))}
    for i in range(len(numerical_features) - 1):
        for j in range(i+1, len(numerical_features)):
            fig, ax = plt.subplots(1,1, figsize=(6,6))
            for _, df in dfs_by_house.items():
                x_label = numerical_features[i]
                y_label = numerical_features[j]
                ax.scatter(df.data[x_label], df.data[y_label])
                plt.xlabel(x_label)
                plt.ylabel(y_label)
            plt.title("Scatter Plot")
            plt.legend([house for house, _ in dfs_by_house.items()])
            plt.show()

if __name__ == '__main__':
    main()
