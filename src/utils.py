import json
import os
import pickle
import re
import sys
from datetime import datetime
from pprint import pprint

import jsonlines
from loguru import logger
from tqdm import auto as tqdm_lib

pprint = pprint


def set_level(level):
    logger.remove()
    logger.add(sys.stdout, level=level)


def get_messages(prompt: str, system_prompt="You are a helpful AI assistant"):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    return messages


def get_chat_prompt(messages, tokenizer):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, think=False)
    return prompt


def get_root():
    nodename = os.uname().nodename
    if nodename == "user-CVN-Z590M-GAMING-PRO":
        return "/sdc"
    else:
        return "/sda"


def extract_code(text: str):
    code_blocks = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)  # output has a complete ```python\n ... \n``` section

    if code_blocks:
        return code_blocks[-1].strip()

    code_blocks = re.findall(r"```python\n(.*)", text, re.DOTALL | re.IGNORECASE)  # output has a ```python\n, but length is not enough
    if code_blocks:
        return code_blocks[-1].strip()

    return "Previous code did not follow the correct format."


def load_pickle(fpath):
    assert str(fpath).endswith(".pkl")
    logger.debug(f"loading pickle file from {fpath}")

    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "rb") as file:
        loaded_data = pickle.load(file)
        return loaded_data


def save_pickle(data, fpath):
    assert str(fpath).endswith(".pkl")
    logger.info(f"saving pickle file to {fpath}")

    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "wb") as file:
        pickle.dump(data, file)


def load_jsonl(fpath, nsamples: int = None):
    assert str(fpath).endswith(".jsonl")

    if nsamples is not None:
        data_iter = jsonlines.open(fpath).iter()
        result = [next(data_iter) for _ in range(nsamples)]
        return result
    else:
        return [item for item in jsonlines.open(fpath).iter()]


def save_jsonl(obj, fpath, mode="w"):
    assert str(fpath).endswith(".jsonl")
    if mode == "w":
        logger.info(f"a jsonl file is saved to: {fpath}")
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with jsonlines.open(fpath, mode) as f:
        f.write_all(obj)


def load_data(fpath):
    if str(fpath).endswith(".jsonl"):
        return load_jsonl(fpath)

    elif str(fpath).endswith(".pkl"):
        return load_pickle(fpath)

    elif str(fpath).endswith(".json"):
        return load_json(fpath)
    else:
        raise NotImplementedError()


def save_data(obj, fpath, mode="w"):
    if str(fpath).endswith(".jsonl"):
        save_jsonl(obj, fpath, mode)

    elif str(fpath).endswith(".pkl"):
        save_pickle(obj, fpath)

    elif str(fpath).endswith(".json"):
        if mode == "a":
            assert isinstance(obj, list)
            if not os.path.exists(fpath):
                save_json(obj, fpath)
            else:
                data = load_json(fpath)
                assert isinstance(data, list)
                save_json(data + obj, fpath)
        else:
            save_json(obj, fpath)

    else:
        raise NotImplementedError()


def load_json(fpath):
    assert str(fpath).endswith(".json")
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(obj: dict, fpath):
    assert str(fpath).endswith(".json")
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4)


def read_file(fpath):
    try:
        with open(fpath, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError as e:
        print(f"{fpath}: {e}")
        return None


def get_existed_ids(field, fout):
    ext = str(fout)
    assert ext.endswith(".jsonl") or ext.endswith(".pkl") or ext.endswith(".json")

    if os.path.exists(fout):
        logger.debug(f"found existed output file {fout}")

        if ext.endswith(".jsonl"):
            existed = load_jsonl(fout)
        if ext.endswith(".pkl"):
            existed = load_pickle(fout)
        if ext.endswith(".json"):
            existed = load_json(fout)
            assert isinstance(existed, list)

        existed_ids = [item[field] for item in existed]
        return existed_ids
    else:
        return None


def filter_data(data: list[dict], field, fout):
    existed_ids = get_existed_ids(field, fout)

    if existed_ids:
        return [item for item in data if item[field] not in existed_ids]
    else:
        return data


def batch_list(lst, k):
    return [lst[i : i + k] for i in range(0, len(lst), k)]


def get_date_time_str():
    now = datetime.now()
    datetime_str = now.strftime("%Y-%m-%d_%H-%M")
    return datetime_str


def setup_proxy():
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


## tqdm
class EmptyTqdm:
    """Dummy tqdm which doesn't do anything."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        self._iterator = args[0] if args else None

    def __iter__(self):
        return iter(self._iterator)

    def __getattr__(self, _):
        """Return empty function."""

        def empty_fn(*args, **kwargs):  # pylint: disable=unused-argument
            return

        return empty_fn

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        return


_tqdm_active = True


class _tqdm_cls:
    def __call__(self, *args, **kwargs):
        if _tqdm_active:
            return tqdm_lib.tqdm(*args, **kwargs)
        else:
            return EmptyTqdm(*args, **kwargs)

    def set_lock(self, *args, **kwargs):
        self._lock = None
        if _tqdm_active:
            return tqdm_lib.tqdm.set_lock(*args, **kwargs)

    def get_lock(self):
        if _tqdm_active:
            return tqdm_lib.tqdm.get_lock()


tqdm = _tqdm_cls()


def is_progress_bar_enabled() -> bool:
    """Return a boolean indicating whether tqdm progress bars are enabled."""
    global _tqdm_active
    return bool(_tqdm_active)


def enable_progress_bar():
    """Enable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = True


def disable_progress_bar():
    """Enable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = False


if __name__ == "__main__":
    data = load_json("outputs/Qwen3-4B-Instruct/lcb_release_v6/zeroshot.json")

    for sample in data:
        print(sample["output_list"][0])
        print("---" * 30)
        # print(sample["code_list"][0])
        # print(extract_code(sample["output_list"][0]))
        print("===" * 30)
