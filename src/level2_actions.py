from py_trees.composites import Composite, Selector, Sequence
from py_trees.decorators import FailureIsSuccess, Retry

from src.common import TaskContext
from src.level1_actions import (
    CodeGenerator,
    AnalogyGenerator,
    PlanGenerator,
    RepairAnalogyGenerator,
    PlanWithAnalogyGenerator,
    StrategyRanker,
    StrategyGuard,
    PopStrategy,
    ReflectionSummarizer,
    RobustRefiner,
)
from src.prompt_utils import PromptBuilder


def get_zs_node(context: TaskContext, num_failures=3) -> Composite:
    gen_seq = Sequence(name="ZS_Gen_Code_Seq", memory=False)
    prompt_node = PromptBuilder("Prompt_Builder", context, mode="zero-shot")
    code_gen_node = CodeGenerator("Code_Generator", context)

    gen_seq.add_children([prompt_node, code_gen_node])
    gen_seq_retry = Retry(name=f"{gen_seq.name}_Retry", child=gen_seq, num_failures=num_failures)

    return gen_seq_retry


def get_cot_node(context: TaskContext, num_failures=3) -> Composite:
    gen_seq = Sequence(name="CoT_Gen_Code_Seq", memory=False)
    prompt_node = PromptBuilder("Prompt_Builder", context, mode="cot")
    code_gen_node = CodeGenerator("Code_Generator", context)

    gen_seq.add_children([prompt_node, code_gen_node])
    gen_seq_retry = Retry(name=f"{gen_seq.name}_Retry", child=gen_seq, num_failures=num_failures)

    return gen_seq_retry


def get_plan_node(context: TaskContext, num_failures=3):
    plan_seq = Sequence(name="Plan_Seq", memory=False)
    plan_gen_node = PlanGenerator(name="Plan_Generator", context=context)
    prompt_node = PromptBuilder("Prompt_Builder", context, mode="plan-to-code")
    code_gen_node = CodeGenerator("Code_Generator", context)

    plan_seq.add_children([plan_gen_node, prompt_node, code_gen_node])
    plan_seq_retry = Retry(name=f"{plan_seq.name}_Retry", child=plan_seq, num_failures=num_failures)

    return plan_seq_retry


def get_analogy_node(context: TaskContext, num_failures=3):
    analogy_seq = Sequence(name="Analogy_Seq", memory=False)
    analogy_gen = AnalogyGenerator(name="Analogy_Generator", context=context)
    prompt_node = PromptBuilder("Prompt_Builder", context, mode="code-with-analogy")
    code_gen_node = CodeGenerator("Code_Generator", context)

    analogy_seq.add_children([analogy_gen, prompt_node, code_gen_node])
    analogy_seq_retry = Retry(name=f"{analogy_seq.name}_Retry", child=analogy_seq, num_failures=num_failures)

    return analogy_seq_retry


def get_self_repair_node(context: TaskContext, num_failures=3):
    repair_seq = Sequence(name="Self_Repair_Seq", memory=False)
    prompt_node = PromptBuilder("Prompt_Builder", context, mode="self-repair")
    code_gen_node = CodeGenerator("Code_Generator", context, mode="repair")

    repair_seq.add_children([prompt_node, code_gen_node])
    repair_seq_retry = Retry(name=f"{repair_seq.name}_Retry_{num_failures}", child=repair_seq, num_failures=num_failures)

    return repair_seq_retry


def get_repair_with_analogy_node(context: TaskContext, num_failures=3):
    repair_seq = Sequence(name="Repair_With_Analogy_Seq", memory=False)
    analogy_gen = RepairAnalogyGenerator(name="Repair_Analogy_Generator", context=context)
    prompt_node = PromptBuilder("Prompt_Builder", context, mode="repair-with-analogy")
    code_gen_node = CodeGenerator("Code_Generator", context, mode="repair")

    repair_seq.add_children([analogy_gen, prompt_node, code_gen_node])
    repair_seq_retry = Retry(name=f"{repair_seq.name}_Retry_{num_failures}", child=repair_seq, num_failures=num_failures)

    return repair_seq_retry


def get_self_repair_with_exp_node(context: TaskContext, num_failures=3):
    repair_seq = Sequence(name="Self_Repair_With_Exp_Seq", memory=False)
    summarizer = ReflectionSummarizer("Reflection_Summarizer", context)
    prompt_node = PromptBuilder("Prompt_Builder", context, mode="self-repair-with-exp")
    code_gen_node = CodeGenerator("Code_Generator", context, mode="repair")

    repair_seq.add_children([summarizer, prompt_node, code_gen_node])
    repair_seq_retry = Retry(name=f"{repair_seq.name}_Retry_{num_failures}", child=repair_seq, num_failures=num_failures)

    return repair_seq_retry


## Level 3: actions


def get_fusion_select_repair_node(context: TaskContext, num_failures=3):
    # Execute all methods sequentially, with self-repair after each failure.
    zs_node = get_zs_node(context, num_failures=1)
    cot_node = get_cot_node(context, num_failures=1)
    analogy_node = get_analogy_node(context, num_failures=1)
    plan_node = get_plan_node(context, num_failures=1)
    nodes = [cot_node, zs_node, analogy_node, plan_node]

    repair_selector = Selector(name="Repair_Selector", memory=True)
    for node in nodes:
        repair_seq = Selector(name=f"{node.name}_Repair_Selector", memory=True)
        self_repair_node = get_self_repair_node(context, num_failures=2)
        repair_seq.add_children([node, self_repair_node])
        repair_selector.add_child(repair_seq)

    repair_selector_retry = Retry(name=f"{repair_selector.name}_Retry", child=repair_selector, num_failures=num_failures)

    return repair_selector_retry


def get_fusion_seq_node(context: TaskContext, num_failures=3):
    # Execute all methods sequentially, saving all successful results
    zs_node = get_zs_node(context, num_failures=num_failures)
    cot_node = get_cot_node(context, num_failures=num_failures)
    analogy_node = get_analogy_node(context, num_failures=num_failures)
    plan_node = get_plan_node(context, num_failures=num_failures)
    self_repair_node = get_self_repair_node(context, num_failures=num_failures)
    nodes = [zs_node, cot_node, analogy_node, plan_node, self_repair_node]

    fusion_seq = Sequence(name="Fusion_Seq", memory=True)

    for node in nodes:
        wrapped_node = FailureIsSuccess(name=f"{node.name}_FIS", child=node)
        fusion_seq.add_child(wrapped_node)

    return fusion_seq


def get_map_coder_node(context: TaskContext, num_failures=3):
    # 1. generate an analogy example for the current problem
    # 2. let the model generate a plan based on the problem and the analogy example
    # 3. generate code based on the analogy example, plan and problem
    # 4. execute the code and perform self-repair based on the analogy example, plan and feedback
    map_coder_selector = Selector(name="MAP_Coder_Selector", memory=True)

    # 1. generate an analogy example for the current problem
    init_seq = Sequence(name="MAP_Coder_Init_Seq", memory=False)
    analogy_gen = AnalogyGenerator(name="Analogy_Generator", context=context)
    plan_gen = PlanWithAnalogyGenerator(name="Plan_With_Analogy_Generator", context=context)
    prompt_code = PromptBuilder(name="Prompt_Builder_Code", context=context, mode="map-coder-code")
    code_gen = CodeGenerator(name="Code_Generator", context=context)
    init_seq.add_children([analogy_gen, plan_gen, prompt_code, code_gen])

    # 2-4
    repair_seq = Sequence(name="MAP_Coder_Repair_Seq", memory=False)
    prompt_repair = PromptBuilder(name="Prompt_Builder_Repair", context=context, mode="map-coder-repair")
    code_repair = CodeGenerator(name="Repair_Code_Generator", context=context, mode="repair")
    repair_seq.add_children([prompt_repair, code_repair])

    repair_retry = Retry(name="Repair_Retry", child=repair_seq, num_failures=num_failures)

    map_coder_selector.add_children([init_seq, repair_retry])
    map_coder_selector_retry = Retry(name=f"{map_coder_selector.name}_Retry", child=map_coder_selector, num_failures=num_failures)

    return map_coder_selector_retry


def get_fusion_dynamic_node(context: TaskContext, num_failures=3):
    # Reorder remaining methods dynamically according to prior execution results, utilizing a predicted strategy selection order as a baseline.
    root_seq = Sequence(name="Dynamic_Strategy_Root_Seq", memory=True)

    # 1. Ranking
    ranker = StrategyRanker(name="Strategy_Ranker", context=context)
    root_seq.add_child(ranker)

    # 2. Dynamic Execution Selector
    # We need to try multiple times until the queue is empty or successful
    # Here we use Retry to wrap the entire Selector, or handle it within the Selector's logic

    dynamic_selector = Selector(name="Dynamic_Priority_Selector", memory=False)

    strategies = {
        "zero-shot": get_zs_node,
        "cot": get_cot_node,
        "plan-to-code": get_plan_node,
        "analogy": get_analogy_node,
    }

    for strategy_name, node_fn in strategies.items():
        branch = Sequence(name=f"{strategy_name}_Branch", memory=False)
        guard = StrategyGuard(name=f"{strategy_name}_Guard", strategy_name=strategy_name, context=context)

        # Core logic: If guard succeeds (is the preferred strategy), try to execute. If execution fails, execute PopStrategy.
        exec_selector = Selector(name=f"{strategy_name}_Exec_Selector", memory=True)
        strategy_node = node_fn(context, num_failures=num_failures)

        repair_after_failure = Sequence(name="Repair_After_Failure", memory=True)
        summarizer = ReflectionSummarizer("Reflection_Summarizer", context)
        repair_retry_seq = Sequence(name="Repair_Retry_Seq", memory=False)
        repair_prompt = PromptBuilder("Repair_Prompt_Builder", context, mode="self-repair-with-exp")
        repair_code_gen = CodeGenerator("Repair_Code_Generator", context, mode="repair")
        repair_retry_seq.add_children([repair_prompt, repair_code_gen])
        repair_retry = Retry(name="Repair_Retry", child=repair_retry_seq, num_failures=num_failures)
        repair_after_failure.add_children([summarizer, repair_retry])

        pop_node = PopStrategy(name=f"{strategy_name}_Pop", context=context)

        if context.use_repair:
            exec_selector.add_children([strategy_node, repair_after_failure, pop_node])
        else:
            exec_selector.add_children([strategy_node, pop_node])

        branch.add_children([guard, exec_selector])
        dynamic_selector.add_child(branch)

    # Wrap the dynamic selector in a retry mechanism to ensure it continues trying until all strategies are exhausted or one succeeds.
    dynamic_retry = Retry(name="Dynamic_Retry", child=dynamic_selector, num_failures=len(strategies))

    root_seq.add_child(dynamic_retry)

    if context.use_robust_guard:
        roubust_node = RobustRefiner(name="Robust_Refiner", context=context)
        robust_refine_retry = Retry(name="Robust_Refine_Retry", child=roubust_node, num_failures=2)
        root_seq.add_child(robust_refine_retry)

    return root_seq
