import numpy as np
import pandas as pd

from eval_helpers import compute_distribution_distance, distribution_distance_names, to_percentage_histogram
from reasonBias.base import get_evaluation_path, get_human_path
from reasonBias.human_helpers import human_labels, human_col_mapping, query_variants

# Description: Computes the effect of switching from CC to Chain in humans.


none_converter_int = lambda x: -1
none_converter_str = lambda x: ""
none_converter = lambda x: None
def float_converter(x):
    try:
        return float(x)
    except Exception:
        return -999999

converters = dict(zip(range(22), [none_converter_int, none_converter_str, none_converter_str, *(18*[float_converter]), none_converter]))

human_col_types = [int, str, str, *(18*[float]), None]

human_data = np.loadtxt(get_human_path() / "Structure_ArgumentsF.csv", delimiter=',', dtype={
    "names": human_labels,
    "formats": human_col_types
}, skiprows=1, converters=converters, quotechar='"', unpack=True)


def get_data(human_data, human_col_name):
    col_data = human_data[human_labels.index(human_col_name)]
    #col_data = col_data[human_data_filter]
    col_data = col_data[col_data != -999999]
    return col_data





def main():
    for metric_name in distribution_distance_names:
        eval_data_base_path = get_evaluation_path() / "human_effect"
        eval_data_base_path.mkdir(parents=True, exist_ok=True)

        data_dump = {
            "query_name": [],
            "cc_to_chain_difference": [],
        }

        for query_name in query_variants: # econ, alien, ...
            human_cc_hist = to_percentage_histogram(get_data(human_data, human_col_mapping[f"cc_{query_name}"]))
            human_chain_hist = to_percentage_histogram(get_data(human_data, human_col_mapping[f"chain_{query_name}"]))

            distance_val = compute_distribution_distance(human_cc_hist, human_chain_hist, metric_name=metric_name)

            data_dump["query_name"].append(query_name)
            data_dump["cc_to_chain_difference"].append(distance_val)

        df = pd.DataFrame.from_dict(data_dump, orient='columns')
        df.to_csv(eval_data_base_path / f"human_effect_{metric_name}.csv", sep=",", index=False)

if __name__ == "__main__":
    main()
