import matplotlib.pyplot as plt
import pandas as pd

from eval_helpers import process_to_percentage_skip_invalid, model_names_from_responses, \
    to_percentage_histogram, compute_distribution_distance, distribution_distance_names
from excerptor.bundle import Bundle
from reasonBias.base import get_query_path, get_evaluation_path

# Description: Computes the effect of switching from cc to chain in LLMs


datasets = [
    "annealing", #cc_ann, chain_ann
    #"top_k_var",
    "top_p_var",
]


def main():
    for metric_name in distribution_distance_names:
        skipped_answers = {}

        eval_data_base_path = get_evaluation_path() / "LLM_effect"
        eval_data_path = eval_data_base_path / f"{metric_name}"
        eval_data_path.mkdir(parents=True, exist_ok=True)

        data_dump = {
            "condition_dataset_name": [],
            "condition_query_name": [],
            "parameter_value": [],
        }

        for dataset_name in datasets:
            print(f"[{dataset_name}] Processing dataset.")

            bundleCC = Bundle(get_query_path() / f"cc_{dataset_name}", name=f"cc_{dataset_name}")
            bundleChain = Bundle(get_query_path() / f"chain_{dataset_name}", name=f"chain_{dataset_name}")
            for queryCC, queryChain in zip(bundleCC.index["queries"], bundleChain.index["queries"]):
                responsesCC = bundleCC.get_all_responses(
                    queryCC["query_id"],
                    {"_query_name": queryCC["meta"]["name"]},
                    proccess_func=process_to_percentage_skip_invalid,
                    meta_object=skipped_answers)
                responsesChain = bundleChain.get_all_responses(
                    queryChain["query_id"],
                    {"_query_name": queryChain["meta"]["name"]},
                    proccess_func=process_to_percentage_skip_invalid,
                    meta_object=skipped_answers)

                query_name = queryCC["meta"]["name"]
                assert queryCC["meta"]["name"] == queryChain["meta"]["name"]

                available_models = model_names_from_responses(responsesCC)

                full_name = f"{dataset_name}_{query_name}"

                plt.figure()

                for j, model_name in enumerate(available_models):
                    variants = Bundle.get_variants(api_names=model_name, meta=bundleCC.index["meta"])
                    var_attr = [k for k, v in bundleCC.index["meta"]["query"].items() if isinstance(v, str)][0]

                    parameter_values = []
                    predicted_values = []

                    for i, variant in enumerate(variants):
                        variant_name = variant["%variant_name"]
                        predictionsCC = responsesCC[variant_name]
                        predictionsChain = responsesChain[variant_name]

                        pred_hist_CC = to_percentage_histogram(predictionsCC)
                        pred_hist_Chain = to_percentage_histogram(predictionsChain)

                        distance_val = compute_distribution_distance(pred_hist_CC, pred_hist_Chain, metric_name=metric_name)
                        parameter_values.append(variant[var_attr])
                        predicted_values.append(distance_val)

                    plt.plot(parameter_values, predicted_values, marker="o", label=model_name)

                    if j == 0:
                        data_dump["condition_dataset_name"].extend([dataset_name] * len(parameter_values))
                        data_dump["condition_query_name"].extend([query_name] * len(parameter_values))
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
