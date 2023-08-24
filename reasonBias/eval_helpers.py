import re

import numpy as np
from scipy.stats import wasserstein_distance, entropy


def model_names_from_responses(responses):
    models = list(set([extract_model_name(k) for k in responses.keys() if not k.startswith("_")]))
    try:
        models.remove("luminous-supreme-control")
    except:
        pass
    return models


def extract_model_name(name):
    parts = name.split("_", 2)
    base = parts[0]
    if parts[1] == "int100query": # FIXME find a better way to do this
        base += "_" + parts[1]
    return base


def process_to_percentage(raw_answer:str, query_id, variant, instance_id, meta_object=None):
    """
    To be used with `bundle.get_all_responses` as `proccess_func` argument.
    :param raw_answer:
    :param query_id:
    :param variant:
    :param instance_id:
    :return:
    """
    raw_answer = raw_answer.strip()

    # some hand coded rules ...
    if raw_answer.endswith("."):
        raw_answer = raw_answer[:-1]
    if raw_answer == "1.0":
        raw_answer = "100"
    #if raw_answer == "4.0":
    #    raw_answer = None

    if not re.match('^[0-9]*$', raw_answer):
        return raw_answer
    return int(raw_answer)


def process_to_percentage_skip_invalid(raw_answer:str, query_id, variant, instance_id, meta_object=None):
    """
    To be used with `bundle.get_all_responses` as `proccess_func` argument.
    :param raw_answer:
    :param query_id:
    :param variant:
    :param instance_id:
    :return:
    """
    raw_answer = raw_answer.strip()

    # some hand coded rules ...
    if raw_answer.endswith("."):
        raw_answer = raw_answer[:-1]
    if raw_answer == "1.0":
        raw_answer = "100"

    value = None
    if re.match('^[0-9]+$', raw_answer):
        value = int(raw_answer)
        if value < 0 or value > 100:
            value = None

    if value is None:
        if meta_object is not None:
            # meta_object collects skipped answers
            if variant not in meta_object:
                meta_object[variant] = []
            meta_object[variant].append(raw_answer)
        return None

    return value


def to_percentage_histogram(predictions):
    hist = np.zeros(101)
    for pred in predictions:
        hist[int(round(pred))] += 1

    count = len(predictions)
    if count != 0:
        hist /= count
    return hist


distribution_distance_names = ["wasserstein", "entropy_diff", "kl_divergence"]


def compute_distribution_distance(distr_a, distr_b, metric_name="wasserstein"):
    if metric_name == "wasserstein":
        distance_val = wasserstein_distance(distr_a, distr_b)
    elif metric_name == "entropy_diff":
        distance_val = abs(entropy(distr_b + 1e-10) - entropy(distr_a + 1e-10))
    elif metric_name == "kl_divergence":
        distance_val = entropy(distr_a + 1e-10, distr_b + 1e-10)  # compute KL
    else:
        raise ValueError(f"Unknown metric name ({metric_name}).")
    return distance_val

