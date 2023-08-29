import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_helpers import process_to_percentage_skip_invalid, model_names_from_responses, \
    to_percentage_histogram, compute_distribution_distance, distribution_distance_names
from excerptor.bundle import Bundle
from reasonBias.base import get_query_path, get_evaluation_path, get_human_path
from reasonBias.human_helpers import human_labels, human_col_mapping

# Description computes the distance between LLM and human answers.

datasets = [
    "cc_annealing",
    "chain_annealing",
    "control_annealing",
    #"cc_top_k_var",
    #"chain_top_k_var",
    #"control_top_k_var",
    "cc_top_p_var",
    "chain_top_p_var",
    "control_top_p_var",
]

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

# filter all rows where Att_Check_18 != 23
human_data_filter = human_data[human_labels.index("Att_Check_18")] == 23


def get_data(human_data, human_col_name):
    col_data = human_data[human_labels.index(human_col_name)]
    #col_data = col_data[human_data_filter]
    col_data = col_data[col_data != -999999]
    return col_data


def main():
    for metric_name in distribution_distance_names:
        skipped_answers = {}

        eval_data_base_path = get_evaluation_path() / "human_fit"
        eval_data_path = eval_data_base_path / f"{metric_name}"
        eval_data_path.mkdir(parents=True, exist_ok=True)

        data_dump = {
            "condition_dataset_name": [],
            "condition_query_name": [],
            "condition_cond": [],
            "parameter_value": [],
        }

        for dataset_name in datasets:
            print(f"[{dataset_name}] Processing dataset.")

            bundle = Bundle(get_query_path() / dataset_name, name=dataset_name)
            for query in bundle.index["queries"]:
                responses = bundle.get_all_responses(
                    query["query_id"],
                    {"_query_name": query["meta"]["name"]},
                    proccess_func=process_to_percentage_skip_invalid,
                    meta_object=skipped_answers)

                query_name = query["meta"]["name"]
                available_models = model_names_from_responses(responses)

                full_name = f"{dataset_name}_{query_name}" #_{bundle.index['meta']['variant']}"

                setup_name = dataset_name.split("_")[0] # "cc" or "chain"
                human_col_name = human_col_mapping[f"{setup_name}_{query_name}"]

                plt.figure()

                for j, model_name in enumerate(available_models):
                    variants = Bundle.get_variants(api_names=model_name, meta=bundle.index["meta"])
                    # '%variant_name', '%pure_variant_name'
                    # find the attribute that is being varied
                    var_attr = [k for k, v in bundle.index["meta"]["query"].items() if isinstance(v, str)][0]

                    parameter_values = []
                    predicted_values = []

                    for i, variant in enumerate(variants):
                        variant_name = variant["%variant_name"]
                        predictions = responses[variant_name]

                        # model_name="gpt-3.5-turbo"|...; variant[var_attr]=0.1|0.2|0.3|...;  predictions = [50, 34, ...]

                        pred_hist = to_percentage_histogram(predictions)
                        human_hist = to_percentage_histogram(get_data(human_data, human_col_name))

                        distance_val = compute_distribution_distance(pred_hist, human_hist, metric_name=metric_name)

                        parameter_values.append(variant[var_attr])
                        predicted_values.append(distance_val)

                    plt.plot(parameter_values, predicted_values, marker="o", label=model_name)

                    if j == 0:
                        #org_parameter_values = parameter_values
                        parts = dataset_name.split("_", 1)
                        data_dump["condition_dataset_name"].extend([parts[0]] * len(parameter_values))
                        data_dump["condition_query_name"].extend([query_name] * len(parameter_values))
                        data_dump["condition_cond"].extend([parts[1]] * len(parameter_values))
                        data_dump["parameter_value"].extend(parameter_values)
                    else:
                        pass
                        # FIXME check all parameter_values are equal (org_parameter_values == parameter_values)

                    if model_name not in data_dump:
                        data_dump[model_name] = []
                    data_dump[model_name].extend(predicted_values)

                plt.legend()
                plt.savefig(eval_data_path / f"plt_{full_name}.jpg")

        #for k,data_dump in data_dumps.items():
        df = pd.DataFrame.from_dict(data_dump, orient='columns')
        df.to_csv(eval_data_base_path / f"data_{metric_name}.csv", sep=",", index_label="idx")

if __name__ == "__main__":
    main()
