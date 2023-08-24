import pandas as pd

from eval_helpers import process_to_percentage_skip_invalid, model_names_from_responses
from excerptor.bundle import Bundle
from reasonBias.base import get_query_path, get_evaluation_path

# Description: dumps all LLM responses per condition (cc/chain_anneal/topp_alien/econ/sex) into CSV format

datasets = [
    "cc_annealing",
    "chain_annealing",
    "control_annealing",
    #"cc_top_k_var",  # including top_k will crash the data_all.csv creation, but the single files can be generated
    #"chain_top_k_var",
    #"control_top_k_var",
    "cc_top_p_var",
    "chain_top_p_var",
    "control_top_p_var"
]

max_answers = 100 # FIXME read from data


def main():
    skipped_answers = {}

    eval_data_base_path = get_evaluation_path() / "data_dump_variants"
    eval_data_base_path.mkdir(parents=True, exist_ok=True)

    full_data_dump = {
        "condition_dataset_name": [],  # cc, chain
        "condition_query_name": [],  # econ,
        "condition_cond": [],  # annealing, topp
        "parameter_value": []
    }

    for dataset_name in datasets:
        print(f"[{dataset_name}] Querying dataset.")

        bundle = Bundle(get_query_path() / dataset_name, name=dataset_name)

        for query in bundle.index["queries"]:
            responses = bundle.get_all_responses(
                query["query_id"],
                {"_query_name": query["meta"]["name"]},
                proccess_func=process_to_percentage_skip_invalid,
                meta_object=skipped_answers)

            query_name = query["meta"]["name"]
            available_models = model_names_from_responses(responses)

            data_dump = {
                "model_name": [],
                "parameter_value": [],
                "predicted_value": [],
            }

            full_name = f"{dataset_name}_{query_name}" #_{bundle.index['meta']['variant']}"

            for j, model_name in enumerate(available_models):
                variants = Bundle.get_variants(api_names=model_name, meta=bundle.index["meta"])
                # '%variant_name', '%pure_variant_name'
                # find the attribute that is being varied
                var_attr = [k for k, v in bundle.index["meta"]["query"].items() if isinstance(v, str)][0]

                if j == 0:
                    parts = dataset_name.split("_", 1)
                    full_data_dump["condition_dataset_name"].extend([parts[0]] * max_answers*len(variants))
                    full_data_dump["condition_query_name"].extend([query_name] * max_answers*len(variants))
                    full_data_dump["condition_cond"].extend([parts[1]] * max_answers*len(variants))

                if model_name not in full_data_dump:
                    full_data_dump[model_name] = []

                for i, variant in enumerate(variants): #  temp/topp = 0.0 ... 2.0
                    variant_name = variant["%variant_name"]
                    predictions = responses[variant_name]

                    data_dump["model_name"].extend([model_name]*len(predictions))
                    data_dump["parameter_value"].extend([variant[var_attr]]*len(predictions))
                    data_dump["predicted_value"].extend(predictions)

                    if j == 0:
                        full_data_dump["parameter_value"].extend([variant[var_attr]] * max_answers)
                    pr_ext = predictions.copy()
                    pr_ext.extend([None] * (max_answers - len(predictions)))
                    full_data_dump[model_name].extend(pr_ext)

            #df = pd.DataFrame.from_dict(data_dump, orient='index').transpose()
            df = pd.DataFrame.from_dict(data_dump, orient='columns')
            df.to_csv(eval_data_base_path / f"data_{full_name}.csv", sep=",", index_label="idx")

        #print("skipped answers:", skipped_answers)

    df = pd.DataFrame.from_dict(full_data_dump, orient='columns')
    df.to_csv(eval_data_base_path / f"data_all.csv", sep=",", index_label="idx")

if __name__ == "__main__":
    main()
