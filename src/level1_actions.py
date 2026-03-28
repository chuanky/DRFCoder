import py_trees
from py_trees.common import Status
from src.common import TaskContext
from src import utils
from . import prompt_utils


logger = utils.logger


def update_tokens(context: TaskContext, action_name: str, response: dict):
    input_tokens = context.input_tokens.get(action_name, 0) + response.get("input_tokens", 0)
    output_tokens = context.output_tokens.get(action_name, 0) + response.get("output_tokens", 0)
    context.input_tokens[action_name] = input_tokens
    context.output_tokens[action_name] = output_tokens


class ReflectionSummarizer(py_trees.behaviour.Behaviour):
    def __init__(self, name, context: TaskContext):
        super(ReflectionSummarizer, self).__init__(name)
        self.client = context.client
        self.context = context
        self.max_tokens = 2048

    def update(self):
        self.context.exp_iter += 1
        if not self.context.generated_code:
            self.context.experience = "No code generated, skipping reflection."
            return Status.SUCCESS

        logger.info(f"--> [Action] Summarizing Experience (Reflection), qid: {self.context.problem.question_id}...")
        problem = self.context.problem
        last_feedback = self.context.test_feedback[-1]["feedback"] if self.context.test_feedback else "No feedback."

        prompt = prompt_utils.get_reflection_prompt(
            problem.question_title,
            problem.question_content,
            self.context.strategy,
            self.context.plan,
            self.context.generated_code,
            str(last_feedback),
            self.context.experience,
            self.context.exp_iter,
        )

        messages = utils.get_messages(prompt)
        response = self.client.complete(messages, self.max_tokens)
        output = response["output"]
        update_tokens(self.context, "reflection", response)
        logger.debug(output)

        self.context.experience = output.strip()
        return Status.SUCCESS


class PlanGenerator(py_trees.behaviour.Behaviour):
    def __init__(self, name, context: TaskContext):
        super(PlanGenerator, self).__init__(name)
        self.client = context.client
        self.context = context
        self.max_tokens = 2048

    def update(self):
        print(f"--> [Action] Generating Plan, qid: {self.context.problem.question_id}...")
        problem = self.context.problem
        prompt = prompt_utils.get_plan_prompt(problem.question_title, problem.question_content, problem.starter_code)
        messages = utils.get_messages(prompt)
        response = self.client.complete(messages, self.max_tokens)
        output = response["output"]
        update_tokens(self.context, "plan_gen", response)
        logger.debug(output)
        self.context.plan = output
        return Status.SUCCESS


class StrategyGenerator(py_trees.behaviour.Behaviour):
    def __init__(self, name, context: TaskContext):
        super(StrategyGenerator, self).__init__(name)
        self.client = context.client
        self.context = context
        self.max_tokens = 1024

    def update(self):
        print(f"--> [Action] Generating Strategy, qid: {self.context.problem.question_id}...")
        messages = utils.get_messages(self.context.prompt)
        response = self.client.complete(messages, self.max_tokens)
        output = response["output"]
        update_tokens(self.context, "strategy_gen", response)
        logger.debug(output)
        self.context.strategy = output
        return Status.SUCCESS


class PlanWithStrategyGenerator(py_trees.behaviour.Behaviour):
    def __init__(self, name, context: TaskContext):
        super(PlanWithStrategyGenerator, self).__init__(name)
        self.client = context.client
        self.context = context
        self.max_tokens = 2048

    def update(self):
        assert self.context.strategy is not None
        print(f"--> [Action] Generating Plan with Strategy, qid: {self.context.problem.question_id}...")
        messages = utils.get_messages(self.context.prompt)
        response = self.client.complete(messages, self.max_tokens)
        output = response["output"]
        update_tokens(self.context, "plan_with_strategy_gen", response)
        logger.debug(output)
        self.context.plan = output
        return Status.SUCCESS


class PlanWithAnalogyGenerator(py_trees.behaviour.Behaviour):
    def __init__(self, name, context: TaskContext):
        super(PlanWithAnalogyGenerator, self).__init__(name)
        self.client = context.client
        self.context = context
        self.max_tokens = 2048

    def update(self):
        assert self.context.analogy is not None
        print(f"--> [Action] Generating Plan with Analogy, qid: {self.context.problem.question_id}...")
        problem = self.context.problem
        prompt = prompt_utils.get_plan_with_analogy_prompt(problem.question_title, problem.question_content, problem.starter_code, self.context.analogy)
        messages = utils.get_messages(prompt)
        response = self.client.complete(messages, self.max_tokens)
        output = response["output"]
        update_tokens(self.context, "plan_with_analogy_gen", response)
        logger.debug(output)
        self.context.plan = output
        return Status.SUCCESS


class AnalogyGenerator(py_trees.behaviour.Behaviour):
    def __init__(self, name, context: TaskContext):
        super(AnalogyGenerator, self).__init__(name)
        self.client = context.client
        self.context = context
        self.max_tokens = 2048

    def update(self):
        logger.info(f"--> [Action] Generating Analogy Example, qid: {self.context.problem.question_id}...")
        problem = self.context.problem
        prompt = prompt_utils.get_analogy_gen_prompt(problem.question_title, problem.question_content, problem.starter_code)
        messages = utils.get_messages(prompt)
        response = self.client.complete(messages, self.max_tokens)
        output = response["output"]
        update_tokens(self.context, "analogy_gen", response)
        logger.debug(output)

        self.context.analogy = output
        return Status.SUCCESS


class RepairAnalogyGenerator(py_trees.behaviour.Behaviour):
    def __init__(self, name, context: TaskContext):
        super(RepairAnalogyGenerator, self).__init__(name)
        self.client = context.client
        self.context = context
        self.max_tokens = 2048

    def update(self):
        if not self.context.generated_code:
            self.context.analogy = "No code generated, skipping repair analogy generation."
            return Status.SUCCESS

        logger.info(f"--> [Action] Generating Repair Analogy Example, qid: {self.context.problem.question_id}...")
        problem = self.context.problem

        last_feedback = "No feedback."
        code = self.context.generated_code
        for feedback in self.context.test_feedback[::-1]:
            if not feedback["passed"]:
                last_feedback = feedback["feedback"]
                code = feedback["code_list"][0]
                break

        prompt = prompt_utils.get_repair_analogy_gen_prompt(problem.question_title, problem.question_content, problem.starter_code, code, last_feedback)
        messages = utils.get_messages(prompt)
        response = self.client.complete(messages, self.max_tokens)
        output = response["output"]
        update_tokens(self.context, "repair_analogy_gen", response)
        logger.debug(output)

        self.context.analogy = output
        return Status.SUCCESS


class CodeGenerator(py_trees.behaviour.Behaviour):
    def __init__(self, name, context: TaskContext, mode="gen", public_only=True):
        super(CodeGenerator, self).__init__(name)
        self.name = name
        self.context = context
        self.client = context.client
        self.mode = mode
        self.public_only = public_only

    def update(self):
        logger.info(f"--> [Action] Generating Code ({self.mode}), qid: {self.context.problem.question_id}...")
        messages = utils.get_messages(self.context.prompt)
        response = self.client.complete(messages, self.context.max_gen_tokens, temperature=0.7)
        output = response["output"]
        update_tokens(self.context, f"code_{self.mode}", response)

        logger.debug(output)
        code = utils.extract_code(output)

        self.context.llm_output = output
        self.context.generated_code = code

        qid = self.context.problem.question_id

        result = self.context.lcb_env.execute(qid, output, public_only=self.public_only)
        self.context.add_feedback(result)

        if result["passed"]:
            print(f"    [Result] {qid} TESTS PASS！")
            if self.context.current_strategy:
                self.context.passed_strategies.append(self.context.current_strategy)
            self.context.passed_codes.append(result["code_list"][0])
            return Status.SUCCESS
        else:
            print(f"    [Result] {qid} TESTS FAIL！")
            return Status.FAILURE


class RobustRefiner(py_trees.behaviour.Behaviour):
    def __init__(self, name, context: TaskContext):
        super(RobustRefiner, self).__init__(name)
        self.client = context.client
        self.context = context
        self.max_tokens = 4096

    def update(self):
        if not self.context.passed_codes:
            logger.warning("No generated code found for robustness refinement.")
            return Status.FAILURE

        logger.info(f"--> [Action] Robust Refine, qid: {self.context.problem.question_id}...")
        problem = self.context.problem
        prompt = prompt_utils.get_robust_refine_prompt(problem.question_title, problem.question_content, problem.starter_code, self.context.passed_codes[-1])

        messages = utils.get_messages(prompt)
        response = self.client.complete(messages, self.max_tokens, temperature=0.7)
        output = response["output"]
        update_tokens(self.context, "robust_refine", response)
        logger.debug(output)

        code = utils.extract_code(output)
        self.context.llm_output = output
        self.context.generated_code = code

        qid = self.context.problem.question_id
        result = self.context.lcb_env.execute(qid, output, public_only=True)
        self.context.add_feedback(result)

        if result["passed"]:
            print(f"    [Result] {qid} ROBUSTNESS REFINEMENT VERSION PASSED PUBLIC TESTS!")
            self.context.passed_codes.append(result["code_list"][0])
            return Status.SUCCESS
        else:
            print(f"    [Result] {qid} ROBUSTNESS REFINEMENT VERSION FAILED PUBLIC TESTS!")
            return Status.FAILURE


class AlwaysFailure(py_trees.behaviour.Behaviour):
    def __init__(self, name="AlwaysFailure"):
        super(AlwaysFailure, self).__init__(name)

    def update(self):
        return Status.FAILURE


class StrategyRanker(py_trees.behaviour.Behaviour):
    def __init__(self, name, context: TaskContext):
        super(StrategyRanker, self).__init__(name)
        self.client = context.client
        self.context = context
        self.max_tokens = 256

    def update(self):
        print(f"--> [Action] Analyzing Problem and Ranking Strategies, qid: {self.context.problem.question_id}...")
        problem = self.context.problem
        prompt = prompt_utils.get_strategy_ranking_prompt(problem.question_title, problem.question_content, problem.starter_code)
        messages = utils.get_messages(prompt)
        response = self.client.complete(messages, self.max_tokens, temperature=0.0)
        output = response["output"]
        update_tokens(self.context, "strategy_ranking", response)
        logger.debug(output)

        # Parse the output into a list of strategy names
        strategies = [s.strip().lower() for s in output.split(",") if s.strip()]
        valid_strategies = self.context.valid_strategies
        ranked_strategies = [s for s in strategies if s in valid_strategies]

        # Ensure all valid strategies are present (fallback)
        for s in valid_strategies:
            if s not in ranked_strategies:
                ranked_strategies.append(s)

        self.context.strategy_queue = ranked_strategies
        self.context.strategy_queue_init = ranked_strategies.copy()
        print(f"    [Result] Strategy Priority Queue: {ranked_strategies}")
        return Status.SUCCESS


class StrategyGuard(py_trees.behaviour.Behaviour):
    def __init__(self, name, strategy_name, context: TaskContext):
        super(StrategyGuard, self).__init__(name)
        self.strategy_name = strategy_name
        self.context = context

    def update(self):
        if self.context.strategy_queue and self.context.strategy_queue[0] == self.strategy_name:
            self.context.current_strategy = self.strategy_name
            return Status.SUCCESS
        return Status.FAILURE


class PopStrategy(py_trees.behaviour.Behaviour):
    def __init__(self, name, context: TaskContext):
        super(PopStrategy, self).__init__(name)
        self.context = context

    def update(self):
        if self.context.strategy_queue:
            strategy = self.context.strategy_queue.pop(0)
            print(f"    [Strategy] Strategy '{strategy}' failed, removing from queue...")
        return Status.FAILURE
