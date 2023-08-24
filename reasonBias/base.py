from pathlib import Path


def get_base_path() -> Path:
    return Path(__file__).parent.resolve()


def get_human_path() -> Path:
    return get_base_path() / "human_data"


def get_query_path() -> Path:
    return get_base_path() / "queries"


def get_evaluation_path() -> Path:
    return get_base_path() / "evaluation"


def get_keys_dir() -> Path:
    return get_base_path() / "keys"
