#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

import argparse
import json
from src import dslr
from src.logistic_regressor import LogisticRegressor

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
        help="describe what append in algorithm")
    parser.add_argument('csvfile', help='dataset_train.csv')
    args = parser.parse_args()

    return args

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
    df.get_numerical_features()
    df.digitalize()
    df.replace_nan()
    df.standardize()
    data = df.standardized
    to_train = {
            "Gryffindor": ["History of Magic", "Transfiguration"],
            "Hufflepuff": df.numerical_features,
            "Ravenclaw": ["Charms", "Muggle Studies"],
            "Slytherin": ["Divination"],
    }
    to_save = {}
    for house, features in to_train.items():
        to_save[house] = {}
        Y = transform_label(data["Hogwarts House"], house)
        X = get_X(data, features)
        logistic_regressor = LogisticRegressor(X, Y)
        if args.verbose:
            print("Training one classifier on class", house)
        logistic_regressor.train(print_cost=args.verbose)
        theta = [logistic_regressor.theta[i] / df.stand_coefs[feature]["sigma"] for
                i, feature in enumerate(features)]
        cte = -sum([df.stand_coefs[feature]["mu"] * logistic_regressor.theta[i] /
            df.stand_coefs[feature]["sigma"] for i, feature in enumerate(features)])
        to_save[house]["cte"] = cte
        for i, feature in enumerate(features):
            to_save[house][feature] = theta[i]
        for feature in df.numerical_features:
            to_save[house].setdefault(feature, 0)

    with open('weights.json', 'w') as outfile:
        json.dump(to_save, outfile)
        outfile.close()

if __name__ == '__main__':
    main()
