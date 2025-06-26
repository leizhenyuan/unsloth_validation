# unsloth_validation
# SFT usage
python run.py --sft --lora --model_name unsloth/Llama-3.2-1B-Instruct --dtype bfloat16 --lora_r 16 --lora_alpha 16 --max_steps 10
python run.py --sft --full_finetune --model_name unsloth/Llama-3.2-1B-Instruct --dtype bfloat16  --max_steps 10
python run.py --sft --qlora --model_name unsloth/Llama-3.2-1B-Instruct --dtype bfloat16  --max_steps 10

# GRPO usage
python run.py --grpo --full_finetune --model_name unsloth/Llama-3.2-1B-Instruct --dtype bfloat16  --max_steps 10
python run.py --grpo --lora --model_name unsloth/Llama-3.2-1B-Instruct --dtype bfloat16  --max_steps 10
python run.py --grpo --qlora --model_name unsloth/Llama-3.2-1B-Instruct --dtype bfloat16  --max_steps 10