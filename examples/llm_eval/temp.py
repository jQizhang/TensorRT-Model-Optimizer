from tensorrt_llm import LLM, SamplingParams

import os

# 1. 绕过 SSH/RSH 依赖，强制使用本地 shell 启动进程
os.environ["OMPI_MCA_plm_rsh_agent"] = "sh"

# 2. 允许以 root 用户运行 (如果你是在 Docker 容器内)
os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"

def main():
    # 指向刚才导出的量化模型路径
    quantized_model_path = "/apps/quant_models/Qwen3-8B-Base-int4-awq-blockwise-qat/hf_checkpoint/"

    # 1. 初始化引擎
    # TensorRT-LLM 会自动检测 config 中的量化字段并构建引擎
    # Disable MPI by setting world_size=1 and force single-process mode
    llm = LLM(
        model=quantized_model_path,
        tensor_parallel_size=1,  # Single GPU
        pipeline_parallel_size=1  # Single stage
    )

    # 2. 准备提示词
    prompts = [
        "Hello, my name is",
        "为什么天空是蓝色的？",
    ]

    # 3. 设置采样参数
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # 4. 运行推理
    outputs = llm.generate(prompts, sampling_params)

    # 5. 打印结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()