from src.utils import keep_only_float, quicksort
from src.math import min, max, mean, std, quartile_n

def get_description(data):
    description = {}
    for field, values in data.items():
        if field == "Index":
            continue
        only_float = keep_only_float(values)
        if len(only_float) > 0:
            float_sorted = quicksort(only_float)
            description.setdefault("Field", []).append(field)
            description.setdefault("Count", []).append(float(len(float_sorted)))
            description.setdefault("Mean", []).append(mean(float_sorted))
            description.setdefault("Std", []).append(std(float_sorted))
            description.setdefault("Min", []).append(min(float_sorted))
            description.setdefault("25%", []).append(quartile_n(float_sorted, 1))
            description.setdefault("50%", []).append(quartile_n(float_sorted, 2))
            description.setdefault("75%", []).append(quartile_n(float_sorted, 3))
            description.setdefault("Max", []).append(max(float_sorted))

    return description

def print_description(description):
    nb_features = len(description["Count"])
    order = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    print("{:<10}".format(""), end="")
    [print("{:>17}".format(description["Field"][n][:15]), end="") for n in range(nb_features)]
    print("")
    for elem in order:
        features = description[elem]
        print("{:<10}".format(elem), end="")
        [print("{:17.6f}".format(round(features[n], 6)), end="") for n in range(nb_features)]
        print("")
    return description
