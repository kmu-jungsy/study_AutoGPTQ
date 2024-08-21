import argparse
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from auto_gptq.utils import Perplexity


from modeling_llamagear import LlamaForCausalLM_GEARKIVI
from UniformAndGroupQuantization import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig
from datasets import load_dataset
import torch
import argparse


if __name__ == "__main__":
    """
    Example usage.

    Default usage with GPT2 model:
    python examples/benchmark/perplexity.py

    Specify GPTQ quantized model:
    python examples/benchmark/perplexity.py \
        --model_name TheBloke/open-llama-7b-open-instruct-GPTQ \
        --model_basename gptq_model-4bit-128g \
        --is_quantized

    Change your dataset:
    python examples/benchmark/perplexity.py --dataset_path tiny_shakespeare

    """
    parser = argparse.ArgumentParser(description="Calculate Perplexity for a model.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b", help="Model name.")
    parser.add_argument("--model_basename", type=str, default=None, help="Model file's basename.")
    parser.add_argument("--n_ctx", type=int, default=512, help="Context size.")
    parser.add_argument("--n_batch", type=int, default=512, help="Batch size.")
    parser.add_argument("--dataset_path", type=str, default="wikitext", help="Path to the dataset.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use.")
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column in the dataset containing the text.",
    )
    parser.add_argument(
        "--per_gpu_max_memory",
        type=int,
        default=None,
        help="Max memory used in each GPU.",
    )
    parser.add_argument("--cpu_max_memory", type=int, default=None, help="Mx memory used in CPU.")
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        help="Whether to use safetensors model file",
    )
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Wheter to use fast tokenizer")
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to use remote code")
    parser.add_argument(
        "--disable_exllama",
        action="store_true",
        help="Whether to use disable exllama kernel",
    )
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    max_memory = {}
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update({i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())})
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    if args.use_safetensors:
        print(
            "The argument --use_safetensors is deprecrated and will be removed in the next release. It is now the default behavior."
        )

    config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

    config.k_bits = 2# current support 2/4 bit for KV Cache
    config.v_bits = 2 # current support 2/4 bit for KV Cache
    config.group_size = 64
    config.residual_length = 64 # the number of recent fp16 tokens

    batch_size = args.n_batch

    ##### Config for 
    compress_config = {}
    compress_config["compress_method"] = "gearlKIVI" # "gearlKIVI" "gearsKIVI"
    compress_config["group_size"] = 64
    compress_config["residual"] = 64
    compress_config["quantize_bit"] = 2
    compress_config["rank"] = 2 ## prefill rank
    compress_config["rankv"] = 2 ## prefill rank
    compress_config["loop"] = 3
    # compress_config["stream_list"] = stream_list
    stream_list = [torch.cuda.Stream(),torch.cuda.Stream()]

    args.model = "None"

    if "gearl" in args.model:
        model = LlamaForCausalLM_GEARKIVI.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            config = config,
            # quantization_config = quantization_config,
            compress_config = compress_config,
            torch_dtype=torch.float16,
            device_map = "cuda:0"
        )
    elif "KIVI" in args.model:
        model = LlamaForCausalLM_KIVI.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            config = config,
            # quantization_config = quantization_config,
            # compress_config = compress_config,
            torch_dtype=torch.float16,
            device_map = "cuda:0"
        )
    elif "None" in args.model:
        model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16,
        device_map = "cuda:0")
    model = model.half()

    max_token = args.n_ctx
    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-2-7b-hf', 
        model_max_length=max_token,
        max_length=max_token,
        use_fast=False, 
        trust_remote_code=True, 
        tokenizer_type='llama')
    tokenizer.pad_token = tokenizer.eos_token

    ppl = Perplexity(
        model,
        tokenizer,
        args.dataset_path,
        args.dataset_name,
        args.split,
        args.text_column,
    )
    ppl.calculate_perplexity(args.n_ctx, args.n_batch)
