import re

import matplotlib.pyplot as plt
import mpmath
import pandas as pd

from excerptor.bundle import Bundle
from excerptor.sequence_helpers import sequence_from_string
from reasonBias.base import get_query_path, get_evaluation_path
from eval_helpers import process_to_percentage_skip_invalid

# Description: Create scatter plots. Not used anymore...


datasets = [
    "cc_annealing",
    "chain_annealing",
    "cc_top_k_var",
    "chain_top_k_var",
    "cc_top_p_var",
    "chain_top_p_var"
]

colors = [
    "red",
    "green",
    "blue",
]

markers = ["x", "o"]


skipped_answers = {}


def main():
    for dataset_name in datasets:
        print(f"[{dataset_name}] Querying dataset.")

        bundle = Bundle(get_query_path() / dataset_name, name=dataset_name)

        #all_responses = []
        for query in bundle.index["queries"]:
            responses = bundle.get_all_responses(
                query["query_id"],
                {"_query_name": query["meta"]["name"]},
                proccess_func=process_to_percentage_skip_invalid,
            meta_object=skipped_answers)

            query_name = query["meta"]["name"]
            available_models = list(set([k.split("_",1)[0] for k in responses.keys() if not k.startswith("_")]))

            data_dump = {
                "model_name": [],
                "parameter_value": [],
                "predicted_value": [],
            }

            full_name = f"{dataset_name}_{query_name}" #_{bundle.index['meta']['variant']}"
            plt.figure(figsize=(15,5))
            plt.title(f"Group: '{full_name}'")
            handles = []

            x_ticks = []

            for j, model_name in enumerate(available_models):
                variants = Bundle.get_variants(api_names=model_name, meta=bundle.index["meta"])
                # '%variant_name', '%pure_variant_name'
                # find the attribute that is being varied
                var_attr = [k for k, v in bundle.index["meta"]["query"].items() if isinstance(v, str)][0]


                #for k, dataset_name in enumerate(datasets):
                #available_models = [k for k in data_dump[dataset_name][0].keys() if not k.startswith("_")]
                #for j, model_name in enumerate(available_models):

                def compute_x0(i, j):
                    return i * (width / len(variants)) + ((j - ((len(available_models)-1)/2)) * m_spacing)

                for i, variant in enumerate(variants):
                    variant_name = variant["%variant_name"]
                    predictions = responses[variant_name]

                    m_spacing = 0.15
                    width = 10.0
                    #spacing = 1.0/(len(available_models)-1)
                    x_0 = compute_x0(i, j)
                    x = [x_0 + (mpmath.rand()-0.5)*0.02 for _ in range(len(predictions))]

                    handle = plt.scatter(
                        x, predictions,
                        c=colors[j],
                        marker=markers[1],
                        alpha=0.2)
                    if i == 0:
                        handles.append([handle, model_name, variant[var_attr]])
                    if j == 0:
                        x_ticks.append([compute_x0(i, (len(available_models)-1)/2), variant[var_attr]])

                    data_dump["model_name"].extend([model_name]*len(predictions))
                    data_dump["parameter_value"].extend([variant[var_attr]]*len(predictions))
                    data_dump["predicted_value"].extend(predictions)

                    plt.boxplot(
                        predictions,
                        widths=0.09,
                        positions=[x_0],
                        #flierprops=dict(markerfacecolor='r', marker='s'),
                        showfliers=False
                    )

            plt.ylim(-5,105)

            plt.xticks([l[0] for l in x_ticks], [l[1] for l in x_ticks])
            #plt.xlim(attr_min - attr_range_pad, attr_max + attr_range_pad)

            plt.ylabel("Prediction")
            plt.xlabel(f"{var_attr.capitalize()}")

            plt.legend(
                [l[0] for l in handles],
                [f"{l[1]}" for l in handles],
                scatterpoints=1,
                loc='lower left',
                ncol=3,
                fontsize=8
            )

            eval_path = get_evaluation_path() / "scatter_variant"
            eval_path.mkdir(parents=True, exist_ok=True)
            eval_data_path = get_evaluation_path() / "scatter_variant_data"
            eval_data_path.mkdir(parents=True, exist_ok=True)

            #plt.savefig(eval_path / f"{full_name}.pdf", dpi=120)
            #plt.savefig(eval_path / f"{full_name}.jpg", dpi=120)
            plt.savefig(eval_path / f"{full_name}.png", dpi=120)
            plt.show()

            #df = pd.DataFrame.from_dict(data_dump, orient='index').transpose()
            df = pd.DataFrame.from_dict(data_dump, orient='columns')
            df.to_csv(eval_data_path / f"data_{full_name}.csv", sep=",", index_label="idx")

        print("skipped answers:", skipped_answers)

if __name__ == "__main__":
    main()
