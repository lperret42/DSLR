#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

from src.logistic_regressor import LogisticRegressor
from src import dslr
from src.utils import parse_arguments, is_float
from src.math import logistic_function, scalar_product
from sklearn.metrics import accuracy_score

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
    df_train, df_test = df.train_test_split()
    df_train.standardize()
    df_train.digitalize()
    df_test.digitalize()
    print("len train:", len(df_train.data["Hogwarts House"]))
    print("len test:", len(df_test.data["Hogwarts House"]))

    df.digitalize()
    df.standardize()
    data = df_train.standardized
    all_features = [feature for feature, values in data.items() if
                                feature != "Index" and is_float(values[0])]
    to_train = {
            "Gryffindor": ["History of Magic", "Transfiguration"],
            "Hufflepuff": all_features,
            "Ravenclaw": ["Charms", "Muggle Studies"],
            "Slytherin": ["Divination"],
            #"Gryffindor": all_features,
            #"Hufflepuff": all_features,
            #"Ravenclaw": all_features,
            #"Slytherin": all_features,
    }
    to_save = {}
    for house, features in to_train.items():
        to_save[house] = {}
        Y = transform_label(data["Hogwarts House"], house)
        X = get_X(data, features)
        logistic_regressor = LogisticRegressor(X, Y)
        logistic_regressor.train(print_cost=False)
        theta = [logistic_regressor.theta[i] / df_train.stand_coefs[feature]["sigma"] for
                i, feature in enumerate(features)]
        cte = -sum([df_train.stand_coefs[feature]["mu"] * logistic_regressor.theta[i] /
            df_train.stand_coefs[feature]["sigma"] for i, feature in enumerate(features)])
        to_save[house]["cte"] = cte
        for i, feature in enumerate(features):
            to_save[house][feature] = theta[i]
        for feature in all_features:
            to_save[house].setdefault(feature, 0)
    nb_error = 0
    Y_true = df_test.data["Hogwarts House"]
    Y_pred = []
    predictions = {}
    for i, real_house in enumerate(df_test.data["Hogwarts House"]):
        x = [df_test.data[feature][i] for feature in all_features]
        for house in list(set(df_test.data["Hogwarts House"])):
            cte = to_save[house]["cte"]
            theta = [to_save[house][feature] for feature in all_features]
            predictions[house] = logistic_function(cte + scalar_product(theta, x))

        print(predictions)
        predict_house = "Gryffindor"
        proba_max = predictions["Gryffindor"]
        for house, proba in predictions.items():
            if proba > proba_max:
                proba_max = proba
                predict_house = house
        Y_pred.append(predict_house)
        if predict_house != real_house:
            nb_error += 1
        #print(predict_house, real_house)

    print("precision:", 1 - nb_error  / len(df_test.data["Hogwarts House"]))
    print(accuracy_score(Y_true, Y_pred))

if __name__ == '__main__':
    main()
