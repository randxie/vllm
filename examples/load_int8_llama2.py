from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir='/home/randxie/vllm/weights',device_map='auto', load_in_8bit=True)