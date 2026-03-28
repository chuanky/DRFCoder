from typing import List, Literal, Optional, Any

from pydantic import BaseModel, Field

from src import utils
from lcb_runner.lm_styles import LMStyle
from lcb_runner.utils.scenarios import Scenario

sys_root = utils.get_root()


class LCBConfig(BaseModel):
    lm_style: LMStyle = LMStyle.CodeQwenInstruct
    trust_remote_code: bool = True
    scenario: Scenario = Scenario.codegeneration
    not_fast: bool = False
    release_version: Literal["release_v2", "release_v6", "demo"] = "release_v6"
    n: int = 1  # Number of samples to generate
    codegen_n: int = 1  # Number of samples for which code generation was run (used to map the code generation file during self-repair)
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    model_max_length: int = 16384
    multiprocess: int = 0  # Number of processes to use for generation (vllm runs do not use this)
    stop: List["str"] = ["###"]
    continue_existing: bool = False
    continue_existing_with_eval: bool = False
    use_cache: bool = False
    cache_batch_size: int = 100
    debug: bool = False
    evaluate: bool = True
    num_process_evaluate: int = 10
    timeout: int = 15
    openai_timeout: int = 90
    tensor_parallel_size: int = 1
    enable_prefix_caching: bool = False
    custom_output_file: Optional[str] = None
    custom_output_save_name: Optional[str] = None
    dtype: str = "bfloat16"
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    @property
    def data_name(self) -> str:
        data_name = f"lcb_{self.release_version}"
        if self.start_date:
            data_name = f"{data_name}-{self.start_date}"
        if self.end_date:
            data_name = f"{data_name}-{self.end_date}"
        return data_name


class TaskContext(BaseModel):
    client: Any
    lcb_env: Any
    problem: Any
    prompt: str = ""
    generated_code: str = ""
    max_gen_tokens: int = 1024
    llm_output: str = ""
    plan: str = ""
    strategy: str = ""
    analogy: str = ""
    plans: List[str] = Field(default_factory=list)
    experience: str = ""
    exp_iter: int = 0
    test_feedback: List[dict] = Field(default_factory=list)
    passed_strategies: List[str] = Field(default_factory=list)
    passed_codes: List[str] = Field(default_factory=list)
    final_result: Optional[dict] = None
    strategy_queue: List[str] = Field(default_factory=list)
    strategy_queue_init: List[str] = Field(default_factory=list)
    current_strategy: Optional[str] = None
    valid_strategies: List[str] = ["zero-shot", "cot", "plan-to-code", "analogy"]
    use_repair: bool = True
    input_tokens: dict[str, int] = Field(default_factory=dict)
    output_tokens: dict[str, int] = Field(default_factory=dict)
    extra_info: dict[str, str] = Field(default_factory=dict)
    use_robust_guard: bool = False
    call_seq: List[str] = Field(default_factory=list)

    def add_feedback(self, feedback: dict):
        self.test_feedback.append(feedback)
        if feedback.get("passed"):
            self.final_result = feedback

    def get_final_result(self):
        for feedback in self.test_feedback:
            if feedback.get("passed"):
                return feedback

        return self.test_feedback[-1]
