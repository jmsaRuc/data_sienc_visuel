import urllib.request
import json


def bits_to_gb(bits):
    return bits / (8 * 1024**3)


def calculate_train_vram_requirements(
        batch_size, seq_len, params, precision, num_layers, num_attn_heads, hidden_size, **ignored
):
    """
    full train, not lora
    source: https://arxiv.org/pdf/2205.05198.pdf (section 4.1)
    credit: https://medium.com/@siddheshgunjal82/understanding-vram-requirements-to-train-inference-with-large-language-models-llms-a3edd0f09d9f
    """
    # Calculate activations using the provided formula
    activations = (
        num_layers * (5/2) * num_attn_heads * batch_size * seq_len**2
                   + 17 * batch_size * hidden_size * seq_len
    )

    # Calculate VRAM using the provided formula
    vram_bits = precision * (activations + params)

    # Convert VRAM from bits to Gigabytes
    return bits_to_gb(vram_bits)


def calculate_inference_vram_requirements(
        batch_size, seq_len, params, precision, num_layers, hidden_size,
        num_attn_heads, num_kv_heads, gqa=True
):
    """
    source 1: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
    source 2: https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
    - same as source 1, but with the introduction a factor (n_heads / n_kv_heads) specific to GQA
      - "GQA helps with keeping the KV cache size down by sharing Keys/Values"
    - defaulting to calculated models using GQA since Mistral, Yi, and Llama 2 use it
    """
    kv_cache = batch_size * seq_len * 2 * num_layers * hidden_size
    if gqa:
        kv_cache *= num_kv_heads / num_attn_heads

    vram_bits = precision * (kv_cache + params)

    return bits_to_gb(vram_bits)


def get_model_params(model_uri):
    url = f"https://huggingface.co/{model_uri}/raw/main/config.json"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def print_table(model_uri, bparams,max_name, batch_size=1,precisions=None, mode="infer"):
    precisions = precisions or [4, 6, 8, 16]

    model_params = get_model_params(model_uri)

    seq_lens = (
        [2**i for i in range(8, 20) if 2**i< model_params[max_name]]
        + [model_params[max_name]]
    )

    calc_params = {
        "num_layers": model_params["num_hidden_layers"],
        "hidden_size": model_params["hidden_size"],
        "num_attn_heads": model_params["num_attention_heads"],
        "num_kv_heads": model_params["num_key_value_heads"],
    }

    if mode == "infer":
        vram_calculator = calculate_inference_vram_requirements
    elif mode == "train":
        vram_calculator = calculate_train_vram_requirements
    elif mode == "train_lora":
        raise NotImplemented
    else:
        raise ValueError

    column_width = 10

    # Print the header of the table with precisions
    header = f"{'SL / BP':>{column_width}}" + "".join([f" | {p:^10}" for p in precisions])
    results = [
        f"Model: {model_uri}",
        f"Params: {bparams}B",
        f"Batch Size: {batch_size}",
        f"Mode: {mode}",
        "",
        "Sequence Length vs Bit Precision - Memory Requirements"
    ]
    results.append(header)
    results.append("-" * len(header))

    # Iterate over each seq_len and calculate VRAM for each precision
    for seq_len in seq_lens:
        seq_len_label = f"{seq_len:>{column_width}}"
        if seq_len == max(seq_lens):
            seq_len_label = "*" + seq_len_label[1:]
        row_data = [seq_len_label]
        for precision in precisions:
            vram_required = vram_calculator(
                batch_size=batch_size,
                seq_len=seq_len,
                precision=precision,
                params=bparams * 1e9,
                **calc_params  # Unpack additional parameters if provided
            )
            row_data.append(f"{vram_required:8.1f}GB")  # Format with 1 decimal point

        # Print each row of the table
        results.append(" | ".join(row_data))

    results += ["", "* Model Max Context Size"]
    results += ["", "Code: https://gist.github.com/lapp0/d28931ebc9f59838800faa7c73e3a0dc/edit"]

    print("    " + "\n    ".join(results))