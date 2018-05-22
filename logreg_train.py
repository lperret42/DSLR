#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

from src.logistic_regressor import LogisticRegressor
from src import dslr
from src.utils import parse_arguments, is_float
import json

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
    df.digitalize()
    df.standardize()
    data = df.standardized
    all_features = [feature for feature, values in data.items() if
                                feature != "Index" and is_float(values[0])]
    to_train = {
            "Gryffindor": ["History of Magic", "Transfiguration"],
            "Hufflepuff": all_features,
            "Ravenclaw": ["Charms", "Muggle Studies"],
            "Slytherin": ["Divination"],
    }
    to_save = {}
    for house, features in to_train.items():
        to_save[house] = {}
        Y = transform_label(data["Hogwarts House"], house)
        X = get_X(data, features)
        logistic_regressor = LogisticRegressor(X, Y)
        logistic_regressor.train(print_cost=False)
        theta = [logistic_regressor.theta[i] / df.stand_coefs[feature]["sigma"] for
                i, feature in enumerate(features)]
        cte = -sum([df.stand_coefs[feature]["mu"] * logistic_regressor.theta[i] /
            df.stand_coefs[feature]["sigma"] for i, feature in enumerate(features)])
        to_save[house]["cte"] = cte
        for i, feature in enumerate(features):
            to_save[house][feature] = theta[i]
        for feature in all_features:
            to_save[house].setdefault(feature, 0)

    with open('weights.json', 'w') as outfile:
        json.dump(to_save, outfile)
        outfile.close()

if __name__ == '__main__':
    main()
