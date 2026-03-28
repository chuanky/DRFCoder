from src import utils


def get_acc(eval_fpath, difficulty="all"):
    eval_results = utils.load_data(eval_fpath)
    if difficulty != "all":
        eval_results = [item for item in eval_results if item["difficulty"] == difficulty]

    corrects = [item for item in eval_results if item["passed"]]
    return f"{len(corrects) / len(eval_results) * 100:.2f}% ({len(eval_results)} sample)"


def get_error_id(fpath, level="all"):
    data = utils.load_data(fpath)

    def process(data, level):
        target_tasks = [item for item in data if item["difficulty"] == level]
        failed_ids = [item["question_id"] for item in target_tasks if not item["passed"]]
        return failed_ids

    if level == "all":
        result = {
            "easy": process(data, "easy"),
            "medium": process(data, "medium"),
            "hard": process(data, "hard"),
        }
    else:
        result = process(data, level)

    print(result)


if __name__ == "__main__":
    fpath = "outputs/Qwen3-9B/lcb_release_v6-2025-02-01/fusion_dynamic_repair_exp_robust_guard_retry_3_run1.jsonl"

    print(get_acc(fpath))
    print("easy:", get_acc(fpath, "easy"))
    print("medium:", get_acc(fpath, "medium"))
    print("hard:", get_acc(fpath, "hard"))
