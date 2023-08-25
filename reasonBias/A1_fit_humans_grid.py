import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_helpers import process_to_percentage_skip_invalid, to_percentage_histogram, compute_distribution_distance, \
    distribution_distance_names
from reasonBias.base import get_evaluation_path, get_human_path
from reasonBias.human_helpers import human_labels, human_col_mapping

num_setups = 3
prompts = {
    "cc": {
        "econ": "43691e8e2eaadca5aa91125efff152108f7e52ada45248808aeeb1c79b4229d4_0",
        "sex": "a839c91dd5b509e30cd463145d808915fda0329f30c8e934931062d65f65e8fc_0",
        "alien": "cdc0627cddead6e4a9d6a7a5f03c875c1856d43ca6396214719eee5b1fb1014e_0",
    },
    "chain": {
        "econ": "0581f76184ee7ee7f0fcf204354d2e21111c06da21063949e54fc4c354246bea_0",
        "sex": "c17422997870075c6bfef5b2600285678c29181f51d03695e0d4002f70c32b5d_0",
        "alien": "00d131c51fa1e764af031c9399ffeb78c24874a3b2f1895183a7512922ef4ff5_0",
    }
}

causal_type_paths = {
    "cc": Path("./queries/cc_grid_topp_temp_g1/instances/"),
    "chain": Path("./queries/chain_grid_topp_temp_g1/instances/")
}

skipped_answers = {}


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


topp_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 1.0]
temp_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

def main():
    num_samples = 100
    models = ["gpt-3.5-turbo", "gpt-4"]

    for metric_name in distribution_distance_names:

        data_dump = {
            "condition_dataset_name": [],  # cc chain
            "condition_query_name": [],  # econ sex alien
            "condition_temp": [],
            "condition_topp": [],
            "parameter_value": [],
        }

        data_dump_samples = {
            "condition_dataset_name": [],  # cc chain
            "condition_query_name": [],  # econ sex alien
            "condition_temp": [],
            "condition_topp": [],
        }

        eval_data_base_path = get_evaluation_path() / "human_grid_fit"
        eval_data_base_path.mkdir(parents=True, exist_ok=True)
        eval_data_np_path = eval_data_base_path / "np"
        eval_data_np_path.mkdir(parents=True, exist_ok=True)

        for j, model_name in enumerate(models):

            data_dump_samples[model_name] = []

            for setup_name, path in causal_type_paths.items(): # cc chain
                for prompt_name, prompt_id in prompts[setup_name].items():  # econ sex alien

                    grid = np.zeros((len(temp_values), len(topp_values)))

                    human_col_name = human_col_mapping[f"{setup_name}_{prompt_name}"]

                    for mm, temp in enumerate(temp_values):
                        for nn, topp in enumerate(topp_values):
                            config_grid_dir = path / prompt_id / f"{model_name}_n{num_samples}_t{temp}_tp{topp}"

                            answers = []
                            for i in range(num_samples):
                                with (config_grid_dir / f"{i}").open("r") as f:
                                    answer = process_to_percentage_skip_invalid(json.load(f)["raw_answer"], None, None, None)
                                    if answer is not None:
                                        answers.append(answer)

                            pred_hist = to_percentage_histogram(answers)
                            human_hist = to_percentage_histogram(get_data(human_data, human_col_name))

                            distance_val = compute_distribution_distance(pred_hist, human_hist, metric_name=metric_name)

                            grid[mm, nn] = distance_val

                            data_dump["condition_dataset_name"].append(setup_name)
                            data_dump["condition_query_name"].append(prompt_name)
                            data_dump["condition_temp"].append(temp)
                            data_dump["condition_topp"].append(topp)
                            data_dump["parameter_value"].append(distance_val)

                            if j == 0:
                                data_dump_samples["condition_dataset_name"].extend([setup_name]*num_samples)
                                data_dump_samples["condition_query_name"].extend([prompt_name]*num_samples)
                                data_dump_samples["condition_temp"].extend([temp]*num_samples)
                                data_dump_samples["condition_topp"].extend([topp]*num_samples)
                            pr_ext = answers.copy()
                            pr_ext.extend([None] * (num_samples - len(answers)))
                            data_dump_samples[model_name].extend(pr_ext)


                    #df = pd.DataFrame.from_dict(data_dump, orient='index').transpose()
                    with (eval_data_np_path / f"data_{metric_name}_{setup_name}_{prompt_name}_{model_name}.np").open("wb+") as f:
                        grid.dump(f)

                    plt.figure()
                    if model_name == "luminous-supreme-control":
                        color = "Purples_r"
                    if model_name == "luminous-supreme-control_int100query":
                        color = "Blues_r"
                    elif model_name == "gpt-3.5-turbo":
                        color = "Oranges_r"
                    elif model_name == "gpt-4":
                        color = "Greens_r"
                    plt.imshow(grid.T, cmap=color)

                    plt.yticks(range(len(topp_values)), topp_values)
                    plt.xticks(range(len(temp_values)), temp_values)
                    plt.ylabel("topp")
                    plt.xlabel("temp")
                    plt.colorbar()

                    plt.savefig(eval_data_base_path / f"plt_{metric_name}_{setup_name}_{prompt_name}_{model_name}.jpg")

        df = pd.DataFrame.from_dict(data_dump, orient='columns')
        df.to_csv(eval_data_base_path / f"data_{metric_name}.csv", sep=",", index_label="idx")

        df = pd.DataFrame.from_dict(data_dump_samples, orient='columns')
        df.to_csv(eval_data_base_path / f"data_{metric_name}_samples.csv", sep=",", index_label="idx")


if __name__ == "__main__":
    main()
