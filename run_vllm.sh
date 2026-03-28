vllm serve /sda/models/Qwen/Qwen3-4B-Instruct-2507 --max-model-len 16384 --gpu-memory-utilization 0.4 --served-model-name Qwen3-4B

vllm serve /sda/models/Qwen3.5-9B --max-model-len 16384 --gpu-memory-utilization 0.7 --served-model-name Qwen3-9B