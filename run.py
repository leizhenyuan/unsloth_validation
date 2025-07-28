import unsloth
import os
import torch
import argparse

from trl import SFTTrainer, GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from transformers import TrainingArguments

def parser_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="LLaMA-3模型微调参数配置")

    # 添加配置参数
    parser.add_argument("--sft", action="store_true", help="是否使用SFT")
    parser.add_argument("--grpo", action="store_true", help="是否使用GRPO")
    parser.add_argument("--full_finetune", action="store_true", help="是否使用full_finetune")
    parser.add_argument("--lora", action="store_true", help="是否使用lora")
    parser.add_argument("--qlora", action="store_true", help="是否使用SFT")

    

    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.2-1B-Instruct", help="HuggingFace模型名称")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="数据类型")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA的秩(r值)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA的alpha值")
    parser.add_argument("--batch_size", type=int, default=2, help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--max_steps", type=int, default=10, help="最大训练步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--vllm", action="store_true", help="是否使用vllm进行推理")

    # 解析参数
    args = parser.parse_args()

    if (args.sft and args.grpo):
        raise ValueError("只能选择一个训练方法：, --sft, --grpo")

    invalid_count = 0
    if (args.sft or args.grpo) and args.full_finetune:
        invalid_count += 1
    if (args.sft or args.grpo) and args.lora:
        invalid_count += 1
    if (args.sft or args.grpo) and args.qlora:
        invalid_count += 1
    
    if invalid_count > 2:
        raise ValueError("只能选择一个训练模式：, --full_finetune, --lora, --qlora")

    return args


def print_xpu_max_memory():
    if torch.xpu.is_available():
        max_memory = torch.xpu.max_memory_allocated('xpu')
        print(f"{{'max_memory': {max_memory}}}")
    elif torch.cuda.is_available():
        # 如果使用CUDA，则获取CUDA的最大内存分配
        print(f"{{'max_memory': {torch.cuda.max_memory_allocated()}}}")
        

def reset_peak_memory_stats():
    if torch.xpu.is_available():
        torch.xpu.reset_peak_memory_stats()
    elif torch.cuda.is_available():
        # 如果使用CUDA，则重置CUDA的峰值内存统计
        torch.cuda.reset_peak_memory_stats()


def sft(args):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit= True if args.qlora else False,  # 使用4bit量化
        full_finetuning = True if args.full_finetune else False,  # 是否进行全量微调
    )

    # 配置LoRA参数
    if args.lora or args.qlora:
        model = FastLanguageModel.get_peft_model(
            model,
            r = args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    # 提示模板和数据处理
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    # 加载数据集
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    reset_peak_memory_stats()
    # 配置训练器
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=5,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16= True if args.dtype == "float16" else False,
            bf16= True if args.dtype == "bfloat16" else False,
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
            report_to="none",
        ),
    )
    trainer.train()
    print_xpu_max_memory()

def grpo(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        load_in_4bit = True if args.qlora else False, # Use 4-bit quantization
        fast_inference = True if args.vllm else False, # Use vLLM for fast inference
        max_lora_rank = args.lora_r,
        gpu_memory_utilization = 0.5, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = args.lora_alpha,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )

    import re
    from datasets import load_dataset, Dataset

    # Load and prep dataset
    SYSTEM_PROMPT = """
    Respond in the following format:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """

    XML_COT_FORMAT = """\
    <reasoning>
    {reasoning}
    </reasoning>
    <answer>
    {answer}
    </answer>
    """

    def extract_xml_answer(text: str) -> str:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

    def extract_hash_answer(text: str) -> str | None:
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    # uncomment middle messages for 1-shot prompting
    def get_gsm8k_questions(split = "train") -> Dataset:
        data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
        data = data.map(lambda x: { # type: ignore
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': extract_hash_answer(x['answer'])
        }) # type: ignore
        return data # type: ignore

    dataset = get_gsm8k_questions()

    # Reward functions
    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]['content'] for completion in completions]
        q = prompts[0][-1]['content']
        extracted_responses = [extract_xml_answer(r) for r in responses]
        print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    def int_reward_func(completions, **kwargs) -> list[float]:
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def count_xml(text) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</answer>\n")[-1])*0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
        return count

    def xmlcount_reward_func(completions, **kwargs) -> list[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [count_xml(c) for c in contents]


    training_args = GRPOConfig(
    # use_vllm = True, # use vLLM for fast inference!
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_torch",
        logging_steps = 1,
        bf16 = True if args.dtype == "bfloat16" else False,
        fp16 = True if args.dtype == "float16" else False,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 8, # Decrease if out of memory
        max_prompt_length = 256,
        max_completion_length = 200,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = args.max_steps,
        save_steps = args.max_steps,
        max_grad_norm = 0.1,
        report_to = "none", # Can use Weights & Biases
        output_dir = "outputs",
    )

    reset_peak_memory_stats()
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args = training_args,
        train_dataset = dataset,
    )

    trainer.train()
    print_xpu_max_memory()

if __name__ == "__main__":
    args = parser_args()
    if args.sft:
        sft(args)
    if args.grpo:
        grpo(args)
