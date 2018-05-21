#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

from src.utils import parse_arguments, get_data, is_float
from src import dslr
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

def scatter_plot_matrix_from_dict(dfs, numerical_features):
    nb_numerical_features = len(numerical_features)
    fig, axes = plt.subplots(nb_numerical_features, nb_numerical_features)
    for i in range(nb_numerical_features):
        for j in range(nb_numerical_features):
            for house, df in dfs.items():
                if i == j:
                    axes[i, j].hist(df.data[numerical_features[i]])
                else:
                    axes[i,j].scatter(df.data[numerical_features[i]], df.data[numerical_features[j]])

            if i == nb_numerical_features-1:
                axes[i,j].set_xlabel(numerical_features[j])
            else:
                axes[i,j].set_xticklabels([])
            if j == 0:
                axes[i,j].set_ylabel(numerical_features[i])
            else:
                axes[i, j].set_yticklabels([])
            if i != nb_numerical_features and j != 0:
                axes[i,j].xaxis.set_ticks_position("bottom")

    fig = axes[0,0].figure
    fig.legend([house for house, _ in dfs.items()])
    fig.suptitle("Pair Plot")
    fig.subplots_adjust(hspace=0.0, wspace=0.0, left=0.08, bottom=0.08, top=0.9, right=0.9 )
    return fig, axes

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
    fig, axes = scatter_plot_matrix_from_dict(dfs_by_house, numerical_features)
    plt.show(fig)

if __name__ == '__main__':
    main()
