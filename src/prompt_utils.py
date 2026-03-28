import py_trees
from py_trees.common import Status
from src.common import TaskContext
from src import utils


logger = utils.logger


def get_zeroshot_prompt(title, content, starter_code):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    instruction = """
## MANDATORY OUTPUT RULES:
1. **NO EXPLANATIONS**: Do not provide any introductory text, apologies, meta-commentary, or post-implementation notes.
2. **NO CHATTER**: Strictly zero conversational filler (e.g., "Sure, I can help," "Here is the code," "I hope this helps").
3. **CODE-ONLY**: Your response must start with a markdown code block (e.g., ```python) and end with the closing triple backticks (```).
4. **NO REDUNDANCY**: Do not repeat the problem statement or requirements in your response.

## IMPLEMENTATION GUIDELINES:
- **Starter Code**: If the user provides "Starter Code", you must include it exactly and complete the implementation within or following it.
- **Self-Contained**: If no starter code is provided, write a complete, runnable script (using stdin/stdout for I/O where appropriate).
- **Quality**: Ensure the code is syntactically perfect, optimized for performance, and adheres to the language's best practices (e.g., PEP 8 for Python).
- **Comments**: Only include comments for extremely complex algorithmic logic. Never include comments that describe what the code is doing in general.
- **Formatting**: Use only markdown code blocks. Do not use bolding or headers outside the code block.

## Task: Generate code for the question and response in specified format.
""".strip()

    prompt = f"""
{question}

{instruction}
""".strip()
    return prompt


def get_cot_prompt(title, content, starter_code):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    instruction = """
## MANDATORY OUTPUT RULES:
1. **THINK FIRST**: You must start your response by analyzing the problem, identifying edge cases, and planning the algorithm.
2. **NO CHATTER**: Strictly zero conversational filler (e.g., "Sure," "Here is the code").
3. **STRUCTURE**: 
   - First, provide your reasoning process.
   - Finally, provide the complete, runnable code inside a single markdown code block (e.g., ```python).
4. **NO REDUNDANCY**: Do not repeat the problem statement.

## IMPLEMENTATION GUIDELINES:
- **Self-Contained**: Write a complete, runnable script using stdin/stdout.
- **Quality**: Ensure the code is syntactically perfect and optimized.
- **Formatting**: Use only markdown code blocks for the code part.

## Task: Generate a step-by-step reasoning and then the implementation.
""".strip()

    prompt = f"""
{question}

{instruction}
""".strip()
    return prompt


def get_plan_prompt(title, content, starter_code):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    instruction = """
## MANDATORY OUTPUT RULES:
1. **PLAN ONLY**: Provide a detailed step-by-step plan to solve the problem. Do not write any code.
2. **NO CHATTER**: Strictly zero conversational filler.
3. **STRUCTURE**: Outline the logic, data structures, and algorithms you will use.

## Task: Generate a detailed plan to solve the problem.
""".strip()

    prompt = f"""
{question}

{instruction}
""".strip()
    return prompt


def get_code_with_plan_prompt(title, content, starter_code, plan):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    plan_section = f"""
## Plan:
{plan}
""".strip()

    instruction = """
## MANDATORY OUTPUT RULES:
1. **ANALYZE FIRST**: Begin your response by providing a concise analysis of the provided plan. Identify key logic, potential edge cases, and the required data structures.
2. **FOLLOW THE PLAN**: Generate code strictly based on the provided plan and your analysis.
3. **CODE BLOCK**: After the analysis, provide the implementation within a single markdown code block (e.g., ```python).
4. **NO CHATTER**: Do not provide introductory conversational filler, post-implementation notes, or "here is the code" style transitions.

## IMPLEMENTATION GUIDELINES:
- **Starter Code**: If provided, include it exactly and complete the implementation.
- **Self-Contained**: If no starter code, write a complete, runnable script.

## Task: Generate code for the question based on the plan.
""".strip()

    prompt = f"""
{question}

{plan_section}

{instruction}
""".strip()
    return prompt


def get_reflection_prompt(title, content, strategy, plan, code, feedback, current_experience, exp_iter):
    prompt = f"""
## Problem:
{title}
{content}

## Attempt Details (Iteration {exp_iter}):
{strategy}

{plan}

### Code:
```python
{code}
```

### Test Feedback:
{feedback}

## Experience:
{current_experience if current_experience else "None."}

## Task: Analysis the failure in **Attempt Details (Iteration {exp_iter})** and strictly follow the **Response Format**.

## Response Format:
    ### History Experience
    [Maintain at most five bullet points which can best summarized the **Experience**]

    ### Current Experience
    [Three bullet points which summarize the failure of **Code** in **Attempt Details (Iteration {exp_iter})** based on **Strategy**, **Plan** and **Test Feedback**.]
""".strip()
    return prompt


def get_repair_prompt(title, content, starter_code, code, feedback):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    previous_attempt = f"""
## Previous Code Attempt:
```python
{code}
```

## Test Feedback:
{feedback}
""".strip()

    instruction = """
## MANDATORY OUTPUT RULES:
1. **ANALYZE FIRST**: You must start your response by analyzing why the previous code failed based on the test feedback.
2. **NO CHATTER**: Strictly zero conversational filler.
3. **STRUCTURE**: 
   - First, provide your error analysis.
   - Finally, provide the complete, fixed, runnable code inside a single markdown code block.
4. **NO REDUNDANCY**: Do not repeat the problem statement.

## Task: Analyze the error and provide the fixed implementation.
""".strip()

    prompt = f"""
{question}

{previous_attempt}

{instruction}
""".strip()
    return prompt


def get_repair_analogy_gen_prompt(title, content, starter_code, code, feedback):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    previous_attempt = f"""
## Previous Code Attempt:
```python
{code}
```

## Test Feedback:
{feedback}
""".strip()

    instruction = """
## MANDATORY OUTPUT RULES:
1. **RECALL**: Recall **two** similar repair scenarios where a piece of code had a specific bug and was fixed. For each scenario, you MUST include:
    - **Bug Description**: What was wrong with the initial code.
    - **Fix Strategy**: How the bug was resolved.
    - **Key Lesson**: Why this fix is relevant to the current failure.

2. **NO CHATTER**: Strictly zero conversational filler. Start directly with the first heading.
3. **STRUCTURE**: 
   - ### Similar Repair Scenarios (1, 2)
   - ### Comparative Repair Analysis

## Task: Recall similar repair scenarios and analyze them to help fix the current code.
""".strip()

    prompt = f"""
{question}

{previous_attempt}

{instruction}
""".strip()
    return prompt


def get_analogy_gen_prompt(title, content, starter_code):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    instruction = """
## MANDATORY OUTPUT RULES:
1. **RECALL**: Recall **three** similar competitive programming problems. For each problem, you MUST include:
    - **Core Mechanism**: The underlying algorithm or data structure used.
    - **Detailed Strategy**: A concise but technical step-by-step solution.
    - **Key Insight**: The specific reason why this problem is analogous to the current one.

2. **NO CHATTER**: Strictly zero conversational filler. Start directly with the first heading.
3. **STRUCTURE**: 
   - ### Similar Problems (1, 2, 3)
   - ### Comparative Analysis

## Task: Recall similar problems and analyze them.
""".strip()

    prompt = f"""
{question}

{instruction}
""".strip()
    return prompt


def get_code_with_analogy_prompt(title, content, starter_code, analogy):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    analogy_section = f"""
## Analogical Reasoning:
{analogy}
""".strip()

    instruction = """
## MANDATORY OUTPUT RULES:
1. **MAPPING & LOGIC**: 
    - Briefly map the key components from the "Analogical Reasoning" to the specific variables and structures of this problem.
    - Explain how the core "Insight" from the analogy handles the constraints of this problem.
    - Limit to 3-5 high-density bullet points.

2. **IMPLEMENTATION**:
    - Provide a **production-ready, optimal** implementation in a single Markdown block (e.g., ```python).
    - **Efficiency**: The code must meet the problem's time/space complexity requirements.
    - **Robustness**: Include handling for edge cases (e.g., empty inputs, single element, maximum constraints).
    - **Readability**: Use meaningful variable names and add concise comments for complex logic.
    - **Language**: The implementation must be python code.

3. **NO CHATTER**: Strictly zero conversational filler. Start directly with the first heading.

4. **STRUCTURE**: 
   - ### Logic Mapping
   - ### Implementation

## Task: Generate optimal code by strictly applying the analogical insights.
""".strip()

    prompt = f"""
{question}

{analogy_section}

{instruction}
""".strip()
    return prompt


def get_repair_with_analogy_prompt(title, content, starter_code, code, feedback, analogy):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    previous_attempt = f"""
## Previous Code Attempt:
```python
{code}
```

## Test Feedback:
{feedback}
""".strip()

    analogy_section = f"""
## Repair Analogical Reasoning:
{analogy}
""".strip()

    instruction = """
## MANDATORY OUTPUT RULES:
1. **ANALYZE & MAP**: 
    - Analyze why the previous code failed based on the test feedback.
    - Map insights from the "Repair Analogical Reasoning" to the current failure.
2. **NO CHATTER**: Strictly zero conversational filler.
3. **STRUCTURE**: 
   - First, provide your joint error and analogical analysis.
   - Finally, provide the complete, fixed, runnable code inside a single markdown code block.

## Task: Use the repair analogies to fix the current code failure.
""".strip()

    prompt = f"""
{question}

{previous_attempt}

{analogy_section}

{instruction}
""".strip()
    return prompt


def get_plan_with_analogy_prompt(title, content, starter_code, analogy):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    analogy_section = f"""
## Analogical Reasoning:
{analogy}
""".strip()

    instruction = """
## MANDATORY OUTPUT RULES:
1. **PLAN ONLY**: Provide a detailed step-by-step plan to solve the problem. Do not write any code.
2. **USE ANALOGY**: Use the insights from the provided "Analogical Reasoning" to inform your plan.
3. **NO CHATTER**: Strictly zero conversational filler.
4. **STRUCTURE**: Outline the logic, data structures, and algorithms you will use.

## Task: Generate a detailed plan to solve the problem using analogical reasoning.
""".strip()

    prompt = f"""
{question}

{analogy_section}

{instruction}
""".strip()
    return prompt


def get_code_with_analogy_and_plan_prompt(title, content, starter_code, analogy, plan):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    analogy_section = f"""
## Analogical Reasoning:
{analogy}
""".strip()

    plan_section = f"""
## Plan:
{plan}
""".strip()

    instruction = """
## MANDATORY OUTPUT RULES:
1. **ANALYZE FIRST**: Begin your response by analyzing how the "Analogical Reasoning" and the "Plan" combine to solve this specific problem.
2. **FOLLOW THE PLAN**: Generate code strictly based on the provided plan and analogical insights.
3. **CODE BLOCK**: Provide the implementation within a single markdown code block (e.g., ```python).
4. **NO CHATTER**: Do not provide introductory conversational filler.

## IMPLEMENTATION GUIDELINES:
- **Starter Code**: If provided, include it exactly.
- **Self-Contained**: If no starter code, write a complete, runnable script.

## Task: Generate optimal code based on the plan and analogical reasoning.
""".strip()

    prompt = f"""
{question}

{analogy_section}

{plan_section}

{instruction}
""".strip()
    return prompt


def get_repair_with_analogy_and_plan_prompt(title, content, starter_code, code, feedback, analogy, plan):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    previous_attempt = f"""
## Previous Code Attempt:
```python
{code}
```

## Test Feedback:
{feedback}
""".strip()

    analogy_section = f"""
## Analogical Reasoning:
{analogy}
""".strip()

    plan_section = f"""
## Plan:
{plan}
""".strip()

    instruction = """
## MANDATORY OUTPUT RULES:
1. **JOINT ANALYSIS**: Analyze the failure based on test feedback, while reconsidering the "Analogical Reasoning" and the original "Plan".
2. **FIX STRATEGY**: Determine if the plan was flawed or if the implementation deviated from it.
3. **STRUCTURE**: 
   - First, provide your joint error, plan, and analogical analysis.
   - Finally, provide the complete, fixed, runnable code inside a single markdown code block.
4. **NO CHATTER**: Strictly zero conversational filler.

## Task: Fix the code failure by integrating feedback with analogical reasoning and the plan.
""".strip()

    prompt = f"""
{question}

{previous_attempt}

{analogy_section}

{plan_section}

{instruction}
""".strip()
    return prompt


def get_strategy_ranking_prompt(title, content, starter_code):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    instruction = """
## Task:
Analyze the programming problem and rank the following strategies based on their likelihood of success (from highest to lowest).

## Strategy Inventory:
1. **zero-shot**: Best for very simple, direct problems with clear logic.
2. **cot**: Best for problems requiring step-by-step logical reasoning or intermediate state tracking.
3. **plan-to-code**: Best for complex problems requiring careful algorithm design or multi-step procedures.
4. **analogy**: Best for problems that resemble classic competitive programming patterns or known algorithms.

## MANDATORY OUTPUT FORMAT:
Output ONLY a comma-separated list of the strategy names in order of preference (e.g., "plan-to-code, cot, zero-shot, analogy"). 
Do NOT include any explanations, bullet points, or extra text.
""".strip()

    prompt = f"""
{question}

{instruction}
""".strip()
    return prompt


def get_repair_with_exp_prompt(title, content, starter_code, code, feedback, experience):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    previous_attempt = f"""
## Previous Code Attempt:
```python
{code}
```

## Test Feedback:
{feedback}

## Experience:
{experience}
""".strip()

    instruction = """
## MANDATORY OUTPUT RULES:
1. **ANALYZE FIRST**: You must start your response by analyzing why the previous code failed based on the test feedback and the "Experience" provided.
2. **STRUCTURE**: 
   - First, provide your joint error and experience analysis. And describe how to correct the "Previous Code Attempt" and avoid the failure in "Test Feedback" and "Experience".
   - Finally, provide the complete, fixed, runnable code inside a single markdown code block. Make sure the fixed implementation avoid the failure in "Test Feedback"  and "Experience".

## Task: Use the "Test Feedback" and "Experience" to provide the fixed implementation.
""".strip()

    prompt = f"""
{question}

{previous_attempt}

{instruction}
""".strip()
    return prompt


def get_strategy_with_exp_prompt(title, content, starter_code, experience):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    exp_section = f"\n## Experience:\n{experience}\n" if experience else ""

    instruction = """
## Response Format:
    ### Strategy:
    [Provide a high-level technical strategy to solve the problem]
""".strip()

    if experience:
        instruction = """
## Response Format:
    ### Experience Analysis:
    [Analyze how the **Experience** can help with providing a high-level strategy for **Question**.]
    
    ### Strategy: 
    [Provide a high-level technical strategy to solve the problem]
"""

    task = "## Task: provide a high-level technical strategy and strictly follow the **Response Format**"

    prompt = f"""
{question}

{exp_section}

{task}

{instruction}
""".strip()
    return prompt


def get_plan_with_strategy_and_exp_prompt(title, content, starter_code, strategy, experience):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    strategy_section = f"""
{strategy}
""".strip()

    exp_section = f"\n## Experience:\n{experience}\n" if experience else ""

    instruction = """
## Response Format:
    ### Plan:
    [Provide a step-by-step plan to solve the problem based on **Strategy**. Do not write any code.]
""".strip()

    if exp_section:
        instruction = """
## Response Format:
    ### Experience Analysis:
    [Analyze how the **Experience** can help with providing a step-by-step plan for **Question**.]

    ### Plan:
    [Provide a detailed step-by-step plan to solve the problem based on **Strategy**. Do not write any code.]
"""

    task = "## Task: Generate a detailed plan to solve the problem using the strategy and strictly follow the **Response Format**."

    prompt = f"""
{question}

{strategy_section}

{exp_section}

{task}

{instruction}
""".strip()
    return prompt


def get_code_with_strategy_plan_and_exp_prompt(title, content, starter_code, strategy, plan, experience):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    strategy_section = f"""
{strategy}
""".strip()

    plan_section = f"""
{plan}
""".strip()

    exp_section = f"\n## Experience:\n{experience}\n" if experience else ""

    instruction = f"""
## Task: Generate optimal code and strictly follow the **Response Format**.

## Response Format:
    ### Analysis
    [Analyzing how the "Strategy", "Plan", {'and " Experience" ' if experience else ""} contribute for solving this problem.]

    ### Implementation
    [Provide the implementation within a single markdown code block (e.g., ```python).]
""".strip()

    prompt = f"""
{question}

{strategy_section}

{plan_section}
{exp_section}
{instruction}
""".strip()
    return prompt


def get_robust_refine_prompt(title, content, starter_code, code):
    question = f"""
## Question:
{title}

{content}
""".strip()
    if starter_code:
        question += f"\n\n## Starter Code:\n```python\n{starter_code}\n```"

    current_code = f"""
## Current Implementation (Passed Public Tests):
```python
{code}
```
""".strip()

    instruction = """
## MANDATORY OUTPUT RULES:
1. **ROBUSTNESS ANALYSIS**: Analyze the current implementation for potential weaknesses, including:
    - **Edge Cases**: Empty input, single element, negative values, null/None, etc.
    - **Boundary Conditions**: Maximum allowed constraints, overflow issues.
    - **Efficiency**: Time and space complexity bottlenecks for large-scale inputs.
    - **Logic Errors**: Subtle bugs that might not be caught by simple test cases.
2. **REFINED IMPLEMENTATION**: Provide a complete, production-ready, and highly robust implementation.
3. **NO CHATTER**: Strictly zero conversational filler.
4. **STRUCTURE**: 
   - First, provide your "Robustness Analysis" which describes how the to refine the implementation to handle potential weaknesses. The description should be concise but technical, ideally in 3-5 bullet points.
   - Finally, provide the refined implementation in a single markdown code block (e.g., ```python).

## Task: Perform a robustness analysis and provide a refined implementation that can pass all possible test cases (including hidden private cases).
""".strip()

    prompt = f"""
{question}

{current_code}

{instruction}
""".strip()
    return prompt


class PromptBuilder(py_trees.behaviour.Behaviour):
    def __init__(self, name, context: TaskContext, mode):
        super(PromptBuilder, self).__init__(name)
        self.name = name
        self.context = context
        self.mode = mode

        if mode == "zero-shot":
            self.prompt_fn = get_zeroshot_prompt
            self.context.max_gen_tokens = 2048
        elif mode == "cot":
            self.prompt_fn = get_cot_prompt
            self.context.max_gen_tokens = 4096
        elif mode == "plan-to-code":
            self.prompt_fn = get_code_with_plan_prompt
            self.context.max_gen_tokens = 2048 * 2
        elif mode == "code-with-analogy":
            self.prompt_fn = get_code_with_analogy_prompt
            self.context.max_gen_tokens = 2048 * 3
        elif mode == "map-coder-code":
            self.prompt_fn = get_code_with_analogy_and_plan_prompt
            self.context.max_gen_tokens = 4096
        elif mode == "exp-strategy":
            self.prompt_fn = get_strategy_with_exp_prompt
            self.context.max_gen_tokens = 1024
        elif mode == "exp-plan":
            self.prompt_fn = get_plan_with_strategy_and_exp_prompt
            self.context.max_gen_tokens = 2048
        elif mode == "exp-code":
            self.prompt_fn = get_code_with_strategy_plan_and_exp_prompt
            self.context.max_gen_tokens = 4096
        elif mode == "self-repair":
            self.prompt_fn = get_repair_prompt
            self.context.max_gen_tokens = 4096
        elif mode == "self-repair-with-exp":
            self.prompt_fn = get_repair_with_exp_prompt
            self.context.max_gen_tokens = 2048 * 3
        elif mode == "repair-with-analogy":
            self.prompt_fn = get_repair_with_analogy_prompt
            self.context.max_gen_tokens = 4096
        elif mode == "map-coder-repair":
            self.prompt_fn = get_repair_with_analogy_and_plan_prompt
            self.context.max_gen_tokens = 4096
        elif mode == "robust-refine":
            self.prompt_fn = get_robust_refine_prompt
            self.context.max_gen_tokens = 4096
        else:
            raise NotImplementedError()

    def update(self):
        logger.info(f"--> [Action] Using {self.mode} mode to generate prompt, qid: {self.context.problem.question_id}...")
        problem = self.context.problem
        self.context.call_seq.append(self.mode)

        if self.mode == "plan-to-code":
            assert self.context.plan is not None
            prompt = self.prompt_fn(problem.question_title, problem.question_content, problem.starter_code, self.context.plan)
        elif self.mode == "code-with-analogy":
            assert self.context.analogy is not None
            prompt = self.prompt_fn(problem.question_title, problem.question_content, problem.starter_code, self.context.analogy)
        elif self.mode == "map-coder-code":
            assert self.context.analogy is not None
            assert self.context.plan is not None
            prompt = self.prompt_fn(problem.question_title, problem.question_content, problem.starter_code, self.context.analogy, self.context.plan)
        elif self.mode == "exp-strategy":
            prompt = self.prompt_fn(problem.question_title, problem.question_content, problem.starter_code, self.context.experience)
        elif self.mode == "exp-plan":
            assert self.context.strategy is not None
            prompt = self.prompt_fn(problem.question_title, problem.question_content, problem.starter_code, self.context.strategy, self.context.experience)
        elif self.mode == "exp-code":
            assert self.context.strategy is not None
            assert self.context.plan is not None
            prompt = self.prompt_fn(
                problem.question_title,
                problem.question_content,
                problem.starter_code,
                self.context.strategy,
                self.context.plan,
                self.context.experience,
            )
        elif self.mode == "robust-refine":
            assert self.context.generated_code != ""
            prompt = self.prompt_fn(problem.question_title, problem.question_content, problem.starter_code, self.context.generated_code)
        elif self.mode == "self-repair" or self.mode == "self-repair-with-exp" or self.mode == "repair-with-analogy" or self.mode == "map-coder-repair":
            last_feedback = "No feedback."
            code = self.context.generated_code
            for feedback in self.context.test_feedback[::-1]:
                if not feedback["passed"]:
                    last_feedback = feedback["feedback"]
                    code = feedback["code_list"][0]
                    break

            if self.mode == "repair-with-analogy":
                assert self.context.analogy is not None
                prompt = self.prompt_fn(problem.question_title, problem.question_content, problem.starter_code, code, last_feedback, self.context.analogy)
            elif self.mode == "map-coder-repair":
                assert self.context.analogy is not None
                assert self.context.plan is not None
                prompt = self.prompt_fn(
                    problem.question_title,
                    problem.question_content,
                    problem.starter_code,
                    code,
                    last_feedback,
                    self.context.analogy,
                    self.context.plan,
                )
            elif self.mode == "self-repair-with-exp":
                prompt = self.prompt_fn(
                    problem.question_title,
                    problem.question_content,
                    problem.starter_code,
                    code,
                    last_feedback,
                    self.context.experience,
                )
            else:
                prompt = self.prompt_fn(problem.question_title, problem.question_content, problem.starter_code, code, last_feedback)
        else:
            prompt = self.prompt_fn(problem.question_title, problem.question_content, problem.starter_code)

        self.context.prompt = prompt
        return Status.SUCCESS
