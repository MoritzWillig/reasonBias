from excerptor.bundle import Bundle
from reasonBias.api.api_helpers import get_lm_by_name
from reasonBias.base import get_query_path, get_keys_dir

dry_run = False
skip_existing = True  # skip samples for which a file exists FIXME DOES NOT WORK AT THE MOMENT (note - i think its working now)!!! delete all existing results and reset idx before running this script ...
test_single = False  # if true, stops after the first query

active_apis = [
    "gpt-3.5-turbo",
    "gpt-4",
    #"luminous-supreme-control", #prompts embedded in template - something changed. prompts are not respected ...
    "luminous-supreme-control_int100query", # instruct to explicitely answer between 0 and 100
    #"luminous-supreme-control_nocontrol", # give the plain prompt - prompts are not respected
]

datasets = [
    "cc_annealing",
    "chain_annealing",
    "control_annealing",
    "cc_top_k_var",
    "chain_top_k_var",
    "control_top_k_var",
    "cc_top_p_var",
    "chain_top_p_var",
    "control_top_p_var",
    "cc_grid_topp_temp_g1",
    "chain_grid_topp_temp_g1"
    #"control_grid_topp_temp_g1"
]

api_instances = {}

def main():
    for dataset_name in datasets:
        print(f"[{dataset_name}] Querying dataset.")

        bundle = Bundle(get_query_path() / dataset_name, name=dataset_name)

        model_params = bundle.index["meta"].get("model", {})

        variants = bundle.get_variants(active_apis, meta=bundle.index["meta"])
        for variant in variants:
            api_name = variant["%api_name"]
            if api_name in api_instances:
                lm = api_instances[api_name]
            else:
                print(f"[{api_name}] Starting up API.")
                lm = get_lm_by_name(api_name)(get_keys_dir(), dry_run=dry_run, **model_params)
                api_instances[api_name] = lm

            variant_name = variant["%variant_name"]
            queries = bundle.get_remaining_queries(variant=variant)
            print(f"[{variant_name}] {len(queries)} remaining queries.")
            for query in queries:
                print(f"[{variant_name}] {query['parameters']['num_samples']} remaining samples.")
                answers = lm.query_variance(query["text"], **query["parameters"])
                #answers = ['75', '85', '75', '75', '75']
                if answers is None:
                    print("No answer")
                    continue

                for answer in answers:
                    print(">>", answer)
                    bundle.add_query_instance(
                        query["id"],
                        variant=variant_name,
                        answer=answer,
                        append_instance=True
                    )

                if test_single:
                    break
            if test_single:
                break
        if test_single:
            break
    print("done")


if __name__ == "__main__":
    main()
