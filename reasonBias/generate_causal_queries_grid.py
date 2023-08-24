from base import get_query_path
from excerptor.bundle import Bundle

datasets = {
    "cc": {
        "queries": [
            ["This is a hypothetical question. Imagine that color loss in Dufflos leads to craving red crystals in Dufflos. On the other hand, color loss in Dufflos leads to joint pain in them. Given the information above, in your opinion, on a scale of 1 to 100, how likely is it that color loss in  Dufflos leads to joint pain in them? Please respond only with one single number and no text.",
             {"name": "alien"}],
            ["This is a hypothetical question. Imagine that more loan defaults lead to high-interest rates on the one hand and more inflation on the other. Given the information above, in your opinion, on a scale of 1 to 100, how likely is it that high-interest rates lead to leads to more inflation? Please respond only with one single number and no text.",
             {"name": "econ"}],
            ["This is a hypothetical question. Imagine that greater profits for criminal organizations lead to criminalizing sex work on the one hand and higher gender-based crime rates on the other. Given the information above, in your opinion, on a scale of 1 to 100, how likely is it that greater profits for criminal organizations lead to leads to higher gender-based crime rates? Please respond only with one single number and no text.",
             {"name": "sex"}]
        ],
        "meta": {
            "model": {
                "max_tokens": 20
            },
            "query": {
                "num_samples": 100
            }
        }
    },
    "chain": {
        "queries": [
            ["This is a hypothetical question. Imagine that color loss in Dufflos leads to craving red crystals in Dufflos. Craving red crystals in Dufflos, in turn, leads to joint pain in them. Given the information above, in your opinion, on a scale of 1 to 100, how likely is it that color loss in Dufflos leads to joint pain in them? Please respond only with one single number and no text.",
             {"name": "alien"}],
            ["This is a hypothetical question. Imagine that high-interest rates lead to more loan defaults, which leads to more inflation. Given the information above, in your opinion, on a scale of 1 to 100, how likely is it that high-interest rates lead to leads to more inflation? Please respond only with one single number and no text.",
             {"name": "econ"}],
            ["This is a hypothetical question. Imagine that Criminalizing sex work leads to greater profits for criminal organizations, which then leads to higher gender-based crime rates. Given the information above, in your opinion, on a scale of 1 to 100, how likely is it that greater profits for criminal organizations lead to leads to higher gender-based crime rates? Please respond only with one single number and no text.",
             {"name": "sex"}]
        ],
        "meta": {
            "model": {
                "max_tokens": 20
            },
            "query": {
                "num_samples": 100
            }
        }
    },
    "control": {
        "queries": [
            [
                # "This is a hypothetical question." "Please respond only with one single number and no text."
                "This is a hypothetical question. Color loss in Dufflos leads to joint pain. On a scale of 1 to 100, how likely is it that Color loss in Dufflos leads to joint pain? Please respond only with one single number and no text.",
                {"name": "alien"}],
            [
                "This is a hypothetical question. More loan defaults lead to more inflation. On a scale of 1 to 100, how likely is it that more defaults lead to more inflation? Please respond only with one single number and no text.",
                {"name": "econ"}],
            [
                "This is a hypothetical question. Greater profits for criminal organizations lead to higher gender-based crime rates. On a scale of 1 to 100, how likely is it that Greater profits for criminal organizations lead to higher gender-based crime rates? Please respond only with one single number and no text.",
                {"name": "sex"}]
        ],
        "meta": {
            "model": {
                "max_tokens": 20
            },
            "query": {
                "num_samples": 100
            }
        }
    }
}

query_metas = {
    "grid_topp_temp_g1": {
        "num_samples": 100,
        "temperature": "seq:0:2:0.1",
        "top_p": "0.01; seq:0.1:1.0:0.1"
    }
}


for dataset_name, dataset_data in datasets.items():
    for meta_name, query_meta, in query_metas.items():
        query_datas = dataset_data["queries"]
        meta = dataset_data["meta"]

        meta["query"] = query_meta
        meta["variant"] = meta_name

        query_strs = []
        metas = []
        for query_data in query_datas:
            if isinstance(query_data, str):
                query_strs.append(query_data)
                metas.append(None)
            else:
                query_strs.append(query_data[0])
                metas.append(query_data[1])

        dataset_config_name = f"{dataset_name}_{meta_name}"
        base = get_query_path() / dataset_config_name

        bundle = Bundle(base, create_new=True, name=dataset_config_name, meta=meta, exists_ok=True)
        bundle.add_queries(query_strs, metas)
        print(f"[{dataset_config_name}] Created.")
