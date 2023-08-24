import pickle
from decimal import Decimal
from itertools import product
from pathlib import Path
import hashlib
import json
from typing import List, Union, Any, Dict, Optional

from excerptor.sequence_helpers import sequence_from_string


class Bundle:

    ARG_ABREVS = {
        "num_samples": "n",
        "temperature": "t",
        "top_k": "tk",
        "top_p": "tp"
    }

    def __init__(self, base: Path, create_new=False, name=None, meta=None, exists_ok=False):
        self.base = base
        self.instances_base = self.base / "instances"
        self.index_path = self.base / "index.json"

        self.arg_abrevs = self.ARG_ABREVS.copy()
        self.arg_abrevs_inv = {v: k for k, v in self.arg_abrevs.items()}

        if create_new:
            self.base.mkdir(parents=True, exist_ok=exists_ok)
            self.instances_base.mkdir(parents=True, exist_ok=True)

            self.index = {
                "name": name,
                "queries": [],
                "meta": meta
            }
            self._store_index()
        else:
            if meta is not None:
                raise ValueError("Argument 'meta' has to be None, when opening an existing package.")

            with self.index_path.open("r") as f:
                self.index = json.load(f)

            if name is not None and name != self.index["name"]:
                raise ValueError(f"Argument 'name' and package name do not match (argument:'{name}', package:'{self.index['name']}')")

        self._id_to_idx = dict()
        for i, query in enumerate(self.index["queries"]):
            self._id_to_idx[query["query_id"]] = i

    @staticmethod
    def bundle_exists(base: Path):
        return base.exists()

    def _store_index(self):
        with self.index_path.open("w") as f:
            json.dump(self.index, f, ensure_ascii=True, indent=2, sort_keys=True)

    def _get_unique_query_id(self, query_str):
        query_hash = hashlib.sha256(query_str.encode('UTF-8')).hexdigest()
        idx = 0
        while True:
            query_id = f"{query_hash}_{idx}"
            for query_obj in self.index["queries"]:
                if query_obj["query_id"] == query_id:
                    idx += 1
                    break
            else:
                break
        return query_id

    def add_queries(self, query_strs: Union[str, List[str]], metas=None):
        if isinstance(query_strs, str):
            query_strs = [query_strs]
            if metas is None:
                metas = [None] * len(query_strs)
        if len(query_strs) != len(metas):
            raise RuntimeError("Length of query_strs and metas does not match")

        for query_str, meta in zip(query_strs, metas):
            query_id = self._get_unique_query_id(query_str)
            query_obj = {
                "query_str": query_str,
                "query_id": query_id,
                "meta": meta,
                "id_counters": {}
            }
            self.index["queries"].append(query_obj)
            self._id_to_idx[query_id] = len(self.index["queries"]) - 1
        self._store_index()

    def _get_instance_path(self, query_id, variant, instance_id):
        return self.instances_base / f"{query_id}" / f"{variant}" / f"{instance_id}"

    def _get_instance_object(self, query_id, variant, instance_id=0):
        instance_path = self._get_instance_path(query_id, variant, instance_id)
        with instance_path.open("r") as f:
            instance_obj = json.load(f)
        return instance_obj

    def add_query_instance(self, query_id: str, variant: str, answer: Any, meta=None, append_instance: bool=False, allow_overwrite: bool=False, instance_id=None):
        if append_instance and allow_overwrite:
            raise RuntimeError("append_instance and allow_overwrite can not be active at the same time.")

        if not allow_overwrite and not append_instance and self.index["queries"][query_id]["id_counter"] != 0:
            raise RuntimeError("query instance does already exist. (set append_instance=True to add multiple instances or overwrite_instance_id to overwrite existing)")

        query_idx = self._id_to_idx[query_id]
        if variant not in self.index["queries"][query_idx]["id_counters"]:
            self.index["queries"][query_idx]["id_counters"][variant] = 0

        if instance_id is None:
            instance_id = self.index["queries"][query_idx]["id_counters"][variant]

        self.index["queries"][query_idx]["id_counters"][variant] += 1
        self._store_index()

        instance_obj = {
            "query_id": query_id,
            "instance_id": instance_id,
            "raw_answer": answer,
            "processed": {},
            "meta": meta
        }

        instance_path = self._get_instance_path(query_id, variant, instance_id)
        instance_path.parent.mkdir(parents=True, exist_ok=True)
        with instance_path.open("w") as f:
            json.dump(instance_obj, f, ensure_ascii=True, indent=2)

    @staticmethod
    def _generate_variants_from_queries_meta(meta):
        keys = []
        meta_keys = []
        for k in meta.keys():
            (meta_keys if k.startswith("%") else keys).append(k)
        keys.sort()  # makes variant naming consistent and deterministic

        ranges = []
        for k in keys:
            values = sequence_from_string(meta[k])
            ranges.append(values)

        def generate_variant(values):
            variant = {}
            name_parts = []
            for key, value in zip(keys, values):
                variant[key] = value
                key_abrev = Bundle.ARG_ABREVS[key] #self.arg_abrevs[key]
                name_parts.append(f"{key_abrev}{value}")
            variant.update({k: meta[k] for k in meta_keys})
            variant["%pure_variant_name"] = "_".join(name_parts)
            return variant

        variants = list(map(generate_variant, product(*ranges)))
        return variants

    @staticmethod
    def get_variants(api_names: List[str], meta=Dict):
        if isinstance(api_names, str):
            api_names = [api_names]
        if meta is None:
            meta = {}
        else:
            meta = meta["query"]

        meta_variants = Bundle._generate_variants_from_queries_meta(meta)
        variants = []
        for api_name in api_names:
            for meta_variant in meta_variants:
                if api_name in meta_variant.get("%except", []):
                    continue

                variant = meta_variant.copy()
                variant["%api_name"] = api_name
                variant["%variant_name"] = f"{api_name}_{variant['%pure_variant_name']}"
                variants.append(variant)
        return variants

    def get_remaining_queries(self, variant: Dict):
        global_query_parameters = self.index["meta"].get("query", {})

        queries = []

        for query_data in self.index["queries"]:
            query_id = query_data["query_id"]
            query_idx = self._id_to_idx[query_id]

            query_parameters = global_query_parameters.copy()
            query_parameters = {k: v for k, v in query_parameters.items() if not k.startswith("%")}

            # overwrite global config with variant specific values
            for par_name, par_val in variant.items():
                if par_name.startswith("%"):
                    continue
                query_parameters[par_name] = par_val

            variant_name = variant["%variant_name"]
            if variant_name in self.index["queries"][query_idx]["id_counters"]:
                existing_samples = self.index["queries"][query_idx]["id_counters"][variant_name]
            else:
                existing_samples = 0
            remaining_samples = max(global_query_parameters["num_samples"] - existing_samples, 0)  # raise error if too many samples?
            query_parameters["num_samples"] = remaining_samples

            if remaining_samples > 0:
                queries.append({
                    "id": query_id,
                    "text": query_data["query_str"],
                    "parameters": query_parameters
                })
        return queries

    def get_all_responses(self, query_id, container=None, proccess_func=None, skip_none=True, meta_object=None):
        query_idx = self._id_to_idx[query_id]

        if meta_object is None:
            additional_args = {}
        else:
            additional_args = {"meta_object": meta_object}

        responses = {} if container is None else container
        for variant, answer_count in self.index["queries"][query_idx]["id_counters"].items():
            answers = []
            for instance_id in range(answer_count):
                instance_obj = self._get_instance_object(query_id, variant, instance_id)

                entry = instance_obj["raw_answer"]
                if proccess_func is not None:
                    entry = proccess_func(entry, query_id, variant, instance_id, **additional_args)
                if not skip_none or entry is not None:
                    answers.append(entry)
            responses[variant] = answers
        return responses

    def rebuild_index_counters(self):
        for query in self.index["queries"]:
            query_id = query["query_id"]
            counters: Dict = query["id_counters"]

            query_base = self.instances_base / query_id

            files = []
            actual = {}
            for item in query_base.glob("*"):
                if item.is_file():
                    files.append(item.name)
                elif item.is_dir():
                    variant_name = item.name
                    variant_path = query_base / variant_name
                    count = len([f for f in variant_path.glob("*") if f.is_file()])
                    actual[variant_name] = count

            counters.clear()
            counters.update(actual)
            self._store_index()

        #return new_variants, missing, files
        #return files
