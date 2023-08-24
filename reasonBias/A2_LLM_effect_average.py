import matplotlib.pyplot as plt
import numpy as np
import pandas

from eval_helpers import distribution_distance_names
from reasonBias.base import get_evaluation_path

datasets = [
    "annealing",
    #"top_k_var",
    "top_p_var",
]

# Note: Run A2_LLM_effect first!

def main():
    for metric_name in distribution_distance_names:

        human_effect_data_path = get_evaluation_path() / "human_effect"
        human_df = pandas.read_csv(human_effect_data_path / f"human_effect_{metric_name}.csv")
        mean_human_effect = np.mean(human_df["cc_to_chain_difference"].values)

        eval_data_path = get_evaluation_path() / "LLM_effect"
        df = pandas.read_csv(eval_data_path / f"data_{metric_name}.csv")
        #idx	condition_dataset_name	condition_query_name	parameter_value	gpt-3.5-turbo	gpt-4	luminous-supreme-control_int100query

        available_models = list(df.keys())[4:]

        for dataset_name in datasets: #anneal, top_p
            plt.figure()

            available_values = sorted(list(set(df.loc[(df['condition_dataset_name'] == dataset_name)]["parameter_value"].values)))

            for model_name in available_models:
                mean_values = []
                for parameter_value in available_values:
                    mean_values.append(np.mean(df.loc[(df['condition_dataset_name'] == dataset_name) & (df['parameter_value'] == parameter_value)][model_name].values))

                plt.plot(available_values, mean_values, marker="o", label=model_name.split("_")[0])

            plt.plot([available_values[0], available_values[-1]], [mean_human_effect, mean_human_effect], label="human_effect")

            plt.legend()
            plt.title(f"Average effect of switching from CC to CHAIN.\n(metric: {metric_name}, parameter: {dataset_name})")
            plt.xlabel(dataset_name)
            plt.ylabel(metric_name)
            plt.savefig(eval_data_path / f"plt_{metric_name}_{dataset_name}_avg.jpg")


if __name__ == "__main__":
    main()
