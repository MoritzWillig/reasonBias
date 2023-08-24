from excerptor.bundle import Bundle
from reasonBias.base import get_query_path

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
    "chain_grid_topp_temp_g1",
    "control_grid_topp_temp_g1"
]


def main():
    for dataset_name in datasets:
        print(f"[{dataset_name}] Rebuilding query counts.")

        bundle = Bundle(get_query_path() / dataset_name, name=dataset_name)
        bundle.rebuild_index_counters()

    print("done")


if __name__ == "__main__":
    main()
