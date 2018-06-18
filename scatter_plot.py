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
    numerical_features = df.numerical_features
    df_standardized = dslr.DataFrame(data=df.standardized)
    dfs_by_house = {house: df_standardized.get_df_filtered({"Hogwarts House": house}) for
            house in list(set(df.data["Hogwarts House"]))}
    print("The 2 features Astronomy and Defense Against the Dark Arts are similar")
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
