import os

from src import utils
from src.common import LCBConfig
from lcb_runner.benchmarks import CodeGenerationProblem, load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

logger = utils.logger


class LCBEnv:
    def __init__(self, args: LCBConfig):
        self.args = args
        cache_file = f"datasets/{args.data_name}.pkl"

        if os.path.exists(cache_file):
            benchmark = utils.load_data(cache_file)
        else:
            benchmark = load_code_generation_dataset(args.release_version, args.start_date, args.end_date)
            utils.save_data(benchmark, cache_file)

        self.benchmark: list[CodeGenerationProblem] = benchmark
        self.benchmark_map = {item.question_id: item for item in self.benchmark}

    def execute(self, qid, llm_response, public_only=False):
        problem = self.get_problem(qid)
        result = {
            "question_id": qid,
            "difficulty": problem.difficulty.value,
            "output_list": [llm_response],
            "code_list": [utils.extract_code(llm_response)],
        }

        if public_only:
            eval_sample = self.benchmark_map[qid].get_public_eval_sample()
        else:
            eval_sample = self.benchmark_map[qid].get_evaluation_sample()

        logger.debug(f"Evaluating {qid}...")
        metrics = codegen_metrics(
            [eval_sample],
            [result["code_list"]],
            num_process_evaluate=self.args.num_process_evaluate,
            timeout=self.args.timeout,
        )
        feedback = metrics[2][0][0]
        passed = metrics[0]["pass@1"] == 1.0

        result.update({"passed": bool(passed), "feedback": feedback})

        return result

    def get_problem(self, qid: str):
        return self.benchmark_map[qid]


# metrics data structure
# [
#     {
#         "pass@1": 1.0,
#         "detail": {
#             "pass@1": {
#                 "0": 1.0
#             }
#         }
#     },
#     {
#         "0": [
#             [
#                 true,
#                 true,
#                 true,
#                 true,
#                 true
#             ]
#         ]
#     },
#     [
#         [
#             "{\"execution time\": 0.017971515655517578}"
#         ]
#     ]
# ]
