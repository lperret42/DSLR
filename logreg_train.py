#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

from src.logistic_regressor import LogisticRegressor
from src import dslr
from src.utils import parse_arguments, is_float
from src.math import logistic_function, scalar_product

def transform_label(Y, label):
    return [1 if y == label else 0 for y in Y]

def get_X(data, features):
    X = []
    for i in range(len(features[0])):
        X.append([data[feature][i] for feature in features])
    return X

def main():
    args = parse_arguments()
    df = dslr.read_csv(args.csvfile)
    df.remove_nan()
    df.standardize()
    data = df.standardized
    numerical_features = [feature for feature, values in data.items() if
                                feature != "Index" and is_float(values[0])]
    Y = data["Hogwarts House"]
    Y = transform_label(Y, "Gryffindor")
    #X = [[x, y] for x, y in zip(data["Potions"], data["Astronomy"])]
    #X = [[x, y, z] for x, y, z in zip(data["Transfiguration"], data["Flying"], data["History of Magic"])]
    X = get_X(data, numerical_features)
    logistic_regressor = LogisticRegressor(X, Y)
    logistic_regressor.train(print_cost=True)
    for i in range(len(data["Hogwarts House"])):
        predict = logistic_function(
            #scalar_product(logistic_regressor.theta, [data["Potions"][i], data["Astronomy"][i]])
            scalar_product(logistic_regressor.theta, [data[feature][i] for feature in numerical_features])
                )
        print(predict, data["Hogwarts House"][i])


if __name__ == '__main__':
    main()
