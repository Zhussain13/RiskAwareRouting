"""Batched generation with vLLM (one model per process, so GPU memory is freed on exit)."""
import time

import config


class VLLM:
    def __init__(self, model: str):
        from vllm import LLM
        t0 = time.time()
        self.name = model
        self.llm = LLM(model=model, dtype=config.VLLM_DTYPE,
                       gpu_memory_utilization=config.VLLM_GPU_UTIL,
                       max_model_len=config.VLLM_MAX_LEN, max_num_seqs=config.VLLM_MAX_SEQS,
                       seed=config.SEED, trust_remote_code=False)
        self.tokenizer = self.llm.get_tokenizer()
        print(f"loaded {model} in {time.time() - t0:.0f}s")

    def _convs(self, prompts, system):
        return [[{"role": "system", "content": system}, {"role": "user", "content": p}]
                for p in prompts]

    def chat(self, prompts, system: str, max_tokens: int):
        """Greedy decoding for a whole batch. Returns list of (text, tokens_in, tokens_out)."""
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=config.SEED)
        outs = self.llm.chat(self._convs(prompts, system), sp, use_tqdm=True)
        return [(o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids))
                for o in outs]

    def single_stream_speed(self, prompts, system: str, max_tokens: int = 128) -> float:
        """Tokens per second for one request at a time: what a single user would see.
        Batched throughput is much higher, but latency is a per-user number."""
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        toks, secs = 0, 0.0
        for p in prompts:
            t0 = time.time()
            o = self.llm.chat(self._convs([p], system), sp, use_tqdm=False)[0]
            secs += time.time() - t0
            toks += len(o.outputs[0].token_ids)
        return toks / max(secs, 1e-9)
