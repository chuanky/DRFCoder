from src.lcb_env import LCBEnv
from src.common import LCBConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from src import utils
from src.client import LocalClient, ApiClient
from src.eval_utils import get_acc

from src import level2_actions
from src.common import TaskContext
import py_trees


def add_metadata(result, context: TaskContext):
    result.update({"usage": {"input_tokens": context.input_tokens, "output_tokens": context.output_tokens}})
    result.update({"passed_strategies": context.passed_strategies})
    result.update({"strategy_queue_init": list(context.strategy_queue_init)})
    result.update({"call_seq": list(context.call_seq)})
    return result


def run_bt(client, strategy_fn, fout_name, num_retry=3, use_robust_guard=False, valid_strategies=None, use_repair=None):
    utils.set_level("INFO")
    args = LCBConfig(start_date="2025-02-01")
    env = LCBEnv(args)
    fout = f"outputs/{client.serve_model_name}/{env.args.data_name}/{fout_name}.jsonl"

    problems = env.benchmark
    existed_ids = utils.get_existed_ids("question_id", fout)
    if existed_ids:
        problems = [item for item in problems if item.question_id not in existed_ids]

    def process(problem):
        task_context = TaskContext(client=client, lcb_env=env, problem=problem)
        if valid_strategies:
            task_context.valid_strategies = valid_strategies
        if use_repair is not None:
            task_context.use_repair = use_repair
        task_context.use_robust_guard = use_robust_guard
        root = strategy_fn(task_context, num_retry)
        tree = py_trees.trees.BehaviourTree(root)
        tree.tick_tock(period_ms=100, stop_on_terminal_state=True)

        if task_context.passed_codes:
            final_eval = None
            for i, code in enumerate(task_context.passed_codes):
                eval_result = env.execute(problem.question_id, f"```python\n{code}\n```")

                if eval_result.get("passed"):
                    eval_result = add_metadata(eval_result, task_context)
                    if i != 0:
                        eval_result.update({"is_robust": True})
                    return eval_result

                if final_eval is None:
                    final_eval = eval_result
            final_eval = add_metadata(final_eval, task_context)
            return final_eval
        else:
            eval_result = task_context.get_final_result()
            eval_result = add_metadata(eval_result, task_context)
            return eval_result

    with ThreadPoolExecutor(max_workers=15) as executor:
        future_to_item = {executor.submit(process, problem): problem for problem in problems}

        for future in utils.tqdm(as_completed(future_to_item), total=len(problems)):
            result = future.result()
            utils.save_data([result], fout, "a")

    print(get_acc(fout))


def dev_bt(client, strategy_fn):
    utils.set_level("DEBUG")
    args = LCBConfig(start_date="2025-02-01")
    env = LCBEnv(args)
    qid = "abc394_c"  # medium
    # qid = "3763"
    # qid = "abc393_d"  # hard
    # qid = "3765"

    problem = env.get_problem(qid)

    task_context = TaskContext(client=client, lcb_env=env, problem=problem)
    task_context.use_robust_guard = True
    root = strategy_fn(task_context, num_failures=1)
    print(py_trees.display.ascii_tree(root))
    # exit()
    tree = py_trees.trees.BehaviourTree(root)
    tree.tick_tock(period_ms=100, stop_on_terminal_state=True)

    print(task_context.passed_codes, len(task_context.passed_codes))
    print(task_context.input_tokens, task_context.output_tokens)

    eval_result = task_context.get_final_result()  # test only on public cases
    eval_result = env.execute(problem.question_id, eval_result["output_list"][0])  # use all test cases
    eval_result = add_metadata(eval_result, task_context)

    print(eval_result)


def run(client):
    if isinstance(client, ApiClient):
        run_bt(client, level2_actions.get_zs_node, "zero_shot_retry_1", 1)
        run_bt(client, level2_actions.get_zs_node, "zero_shot_retry_1_run2", 1)
        run_bt(client, level2_actions.get_cot_node, "cot_retry_1", 1)
        run_bt(client, level2_actions.get_cot_node, "cot_retry_1_run2", 1)
        run_bt(client, level2_actions.get_cot_node, "cot_retry_3_run1", 3)
        run_bt(client, level2_actions.get_cot_node, "cot_retry_3_run2", 3)
        run_bt(client, level2_actions.get_analogy_node, "analogy_retry_1", 1)
        run_bt(client, level2_actions.get_analogy_node, "analogy_retry_1_run2", 1)
        run_bt(client, level2_actions.get_plan_node, "plan_to_code_retry_1", 1)
        run_bt(client, level2_actions.get_plan_node, "plan_to_code_retry_1_run2", 1)

        run_bt(client, level2_actions.get_self_repair_node, "self_repair_retry_3", 3)
        run_bt(client, level2_actions.get_self_repair_node, "self_repair_retry_3_run2", 3)
        run_bt(client, level2_actions.get_self_repair_with_exp_node, "self_repair_with_exp_retry_3", 3)
        run_bt(client, level2_actions.get_self_repair_with_exp_node, "self_repair_with_exp_retry_3_run2", 3)

        run_bt(client, level2_actions.get_fusion_dynamic_node, "fusion_dynamic_repair_exp_robust_guard_retry_3", 2, use_robust_guard=True)
        run_bt(client, level2_actions.get_fusion_dynamic_node, "fusion_dynamic_repair_exp_robust_guard_retry_3_run2", 2, use_robust_guard=True)
        run_bt(client, level2_actions.get_fusion_dynamic_node, "fusion_dynamic_repair_exp_robust_guard_retry_3_run3", 3, use_robust_guard=True)
        run_bt(client, level2_actions.get_map_coder_node, "map_coder_retry_3", num_retry=3)
        run_bt(client, level2_actions.get_map_coder_node, "map_coder_retry_3_run2", num_retry=2)

    else:
        run_bt(client, level2_actions.get_zs_node, "zero_shot_retry_3")
        run_bt(client, level2_actions.get_zs_node, "zero_shot_retry_3_run2")
        run_bt(client, level2_actions.get_zs_node, "zero_shot_retry_3_run3")
        run_bt(client, level2_actions.get_cot_node, "cot_retry_3")
        run_bt(client, level2_actions.get_cot_node, "cot_retry_3_run2")
        run_bt(client, level2_actions.get_cot_node, "cot_retry_3_run3")
        run_bt(client, level2_actions.get_cot_node, "cot_retry_12", 12)
        run_bt(client, level2_actions.get_analogy_node, "analogy_retry_3")
        run_bt(client, level2_actions.get_analogy_node, "analogy_retry_3_run2")
        run_bt(client, level2_actions.get_analogy_node, "analogy_retry_3_run3")
        run_bt(client, level2_actions.get_plan_node, "plan_to_code_retry_3")
        run_bt(client, level2_actions.get_plan_node, "plan_to_code_retry_3_run2")
        run_bt(client, level2_actions.get_plan_node, "plan_to_code_retry_3_run3")

        run_bt(client, level2_actions.get_self_repair_node, "self_repair_retry_3_run1")
        run_bt(client, level2_actions.get_self_repair_node, "self_repair_retry_3_run2")
        run_bt(client, level2_actions.get_self_repair_node, "self_repair_retry_3_run3")
        run_bt(client, level2_actions.get_self_repair_with_exp_node, "self_repair_with_exp_retry_3")
        run_bt(client, level2_actions.get_self_repair_with_exp_node, "self_repair_with_exp_retry_3_run2")
        run_bt(client, level2_actions.get_self_repair_with_exp_node, "self_repair_with_exp_retry_3_run3")

        run_bt(client, level2_actions.get_fusion_dynamic_node, "fusion_dynamic_repair_exp_robust_guard_retry_3_run1", use_robust_guard=True)
        run_bt(client, level2_actions.get_fusion_dynamic_node, "fusion_dynamic_repair_exp_robust_guard_retry_3_run2", use_robust_guard=True)
        run_bt(client, level2_actions.get_fusion_dynamic_node, "fusion_dynamic_repair_exp_robust_guard_retry_3_run3", use_robust_guard=True)
        run_bt(client, level2_actions.get_map_coder_node, "map_coder_retry_3", num_retry=3)
        run_bt(client, level2_actions.get_map_coder_node, "map_coder_retry_3_run2", num_retry=3)
        run_bt(client, level2_actions.get_map_coder_node, "map_coder_retry_6", num_retry=6)


def ablation():
    strategy_fn = level2_actions.get_fusion_dynamic_node
    client = LocalClient()

    fout = "ablation/fusion_dynamic_repair_exp_retry_3_remove_zeroshot"
    run_bt(client, strategy_fn, fout, use_robust_guard=True, valid_strategies=["cot", "plan-to-code", "analogy"])
    fout = "ablation/fusion_dynamic_repair_exp_retry_3_remove_cot"
    run_bt(client, strategy_fn, fout, use_robust_guard=True, valid_strategies=["zero-shot", "plan-to-code", "analogy"])
    fout = "ablation/fusion_dynamic_repair_exp_retry_3_remove_plan"
    run_bt(client, strategy_fn, fout, use_robust_guard=True, valid_strategies=["zero-shot", "cot", "analogy"])
    fout = "ablation/fusion_dynamic_repair_exp_retry_3_remove_analogy"
    run_bt(client, strategy_fn, fout, use_robust_guard=True, valid_strategies=["zero-shot", "cot", "plan-to-code"])

    run_bt(client, strategy_fn, "ablation/fusion_dynamic_repair_exp_retry_3_only_zeroshot", use_robust_guard=True, valid_strategies=["zero-shot"])
    run_bt(client, strategy_fn, "ablation/fusion_dynamic_repair_exp_retry_3_only_cot", use_robust_guard=True, valid_strategies=["cot"])
    run_bt(client, strategy_fn, "ablation/fusion_dynamic_repair_exp_retry_3_only_plan", use_robust_guard=True, valid_strategies=["plan-to-code"])
    run_bt(client, strategy_fn, "ablation/fusion_dynamic_repair_exp_retry_3_only_analogy", use_robust_guard=True, valid_strategies=["analogy"])

    run_bt(client, level2_actions.get_fusion_dynamic_node, "ablation/fusion_dynamic_remove_repair_exp_retry_3", use_robust_guard=True, use_repair=False)


if __name__ == "__main__":
    # mode = "debug"
    # mode = "ablation"
    mode = "run"

    # client = LocalClient()
    client = LocalClient("/sda/models/Qwen3.5-9B", "Qwen3-9B")
    # client = ApiClient()

    if mode == "debug":
        dev_bt(client, level2_actions.get_fusion_dynamic_node)

    if mode == "run":
        run(client)

    if mode == "ablation":
        ablation()
