import matplotlib.pyplot as plt
import numpy as np
import pandas

from eval_helpers import distribution_distance_names
from reasonBias.base import get_evaluation_path

datasets = ["cc", "chain", "control"]

conditions = [
    "annealing",
    #"top_k_var",
    "top_p_var",
]

# Description computes the distance between LLM and human answers - averaged over conditions
# Note: Run A1_fit_humans first


def main():
    for metric_name in distribution_distance_names:

        human_fit_data_path = get_evaluation_path() / "human_fit"
        df = pandas.read_csv(human_fit_data_path / f"data_{metric_name}.csv")
        # idx	condition_dataset_name	condition_query_name	condition_cond	parameter_value	luminous-supreme-control_int100query	gpt-4	gpt-3.5-turbo

        available_models = list(df.keys())[5:]

        for condition in conditions: #anneal, top_p
            for dataset_name in datasets: #cc, chain
                plt.figure()

                available_values = sorted(list(set(df.loc[((df['condition_dataset_name'] == dataset_name) & (df['condition_cond'] == condition))]["parameter_value"].values)))

                for model_name in available_models:
                    mean_values = []
                    for parameter_value in available_values:
                        mean_values.append(np.mean(df.loc[(df['condition_dataset_name'] == dataset_name) & (df['condition_cond'] == condition) & (df['parameter_value'] == parameter_value)][model_name].values))

                    plt.plot(available_values, mean_values, marker="o", label=model_name.split("_")[0])

                plt.legend()
                plt.title(f"Average distance between human an LLM distributions.\n(metric: {metric_name}, parameter: {condition}, setting: {dataset_name})")
                plt.xlabel(condition)
                plt.ylabel(metric_name)
                plt.savefig(human_fit_data_path / f"plt_{metric_name}_{condition}_{dataset_name}_avg.jpg")


if __name__ == "__main__":
    main()
