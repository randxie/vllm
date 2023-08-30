import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm.engine.arg_utils import EngineArgs, ModelConfig
from vllm.worker.worker import _init_distributed_environment

from vllm.model_executor import get_model, InputMetadata, set_random_seed
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=1)

engine_args = EngineArgs(model="meta-llama/Llama-2-7b-chat-hf", dtype="int8")
model_config, _, parallel_config, _ = engine_args.create_engine_configs()

print("initializing models")
# This env var set by Ray causes exceptions with graph building.
os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
# Env vars will be set by Ray.
rank = 0
local_rank = 0
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Initialize the distributed environment.
distributed_init_method = f"tcp://localhost:10086"
_init_distributed_environment(parallel_config, rank, distributed_init_method)

# Initialize the model.
set_random_seed(model_config.seed)
model = get_model(model_config)

# -----------------------------------------------------------------
# Run LLM engine e2e
# -----------------------------------------------------------------
# Create an LLM.
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    dtype="int8",
    download_dir='/home/randxie/vllm/weights',
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
