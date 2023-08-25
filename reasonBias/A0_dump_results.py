import json

from excerptor.bundle import Bundle
from reasonBias.base import get_query_path, get_evaluation_path
from reasonBias.eval_helpers import process_to_percentage_skip_invalid

# Description: dumps all LLM responses into a single JSON file


datasets_s = {
    "": [
        "cc_annealing",
        "chain_annealing",
        "control_annealing",
        "cc_top_k_var",
        "chain_top_k_var",
        "control_top_k_var",
        "cc_top_p_var",
        "chain_top_p_var",
        "control_top_p_var"
    ],
    "grid_": [
        "cc_grid_topp_temp_g1",
        "chain_grid_topp_temp_g1"
    ]
}


def main():
    for set_id, datasets in datasets_s.items():
        all_skipped_answers = {}
        response_counts_dump = {}
        data_dump = {}

        for dataset_name in datasets:
            print(f"[{dataset_name}] Querying dataset.")
            skipped_answers = {}

            bundle = Bundle(get_query_path() / dataset_name, name=dataset_name)

            all_responses = []
            response_counts = []
            for query in bundle.index["queries"]:
                responses = bundle.get_all_responses(
                    query["query_id"],
                    {"query_name": bundle.index["name"]},
                    proccess_func=process_to_percentage_skip_invalid,
                    meta_object=skipped_answers)
                responses["query_id"] = query["query_id"]
                all_responses.append(responses)

                response_counts.append({k: len(v) if isinstance(v, list) else v for k, v in responses.items()})

            data_dump[dataset_name] = all_responses
            response_counts_dump[dataset_name] = response_counts

            all_skipped_answers[dataset_name] = {k:list(set(v)) for k,v in skipped_answers.items()}
            print("skipped answers:", all_skipped_answers[dataset_name])

        eval_path = get_evaluation_path() / "data_dump"
        eval_path.mkdir(parents=True, exist_ok=True)

        with (eval_path / f"data_dump_{set_id}.json").open("w") as f:
            json.dump(data_dump, f, ensure_ascii=True, indent=2)
        with (eval_path / f"data_dump_{set_id}skipped_answers.json").open("w") as f:
            json.dump(all_skipped_answers, f, ensure_ascii=True, indent=2)
        with (eval_path / f"data_dump_{set_id}response_counts.json").open("w") as f:
            json.dump(response_counts_dump, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
