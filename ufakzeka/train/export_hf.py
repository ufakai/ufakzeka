"""Export a ufakzeka checkpoint to a Hugging Face transformers LlamaForCausalLM folder.

QK-Norm is not in transformers Llama, so the export uses the Qwen3 layout (q_norm and
k_norm per head, otherwise identical: RMSNorm, SwiGLU, RoPE, GQA, tied embeddings).
Logit soft-capping is not in Qwen3 and matters a lot for a model trained with it
(0.27 nats on held out text without it), so the folder ships a small model class that
adds it, loaded with trust_remote_code=True. The script verifies parity after saving."""

from __future__ import annotations

import argparse
import json
import os

import torch


def export(ckpt_path: str, tokenizer_path: str, out_dir: str):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from ufakzeka.train.model import ModelConfig, Transformer

    _raw = Tokenizer.from_file(tokenizer_path)

    def _id(t):
        i = _raw.token_to_id(t)
        assert i is not None, t
        return i

    ck = torch.load(ckpt_path, map_location="cpu")
    mc = ModelConfig(**ck["cfg"]["model"])
    model = Transformer(mc)
    model.load_state_dict(ck["model"])

    hf_cfg = Qwen3Config(
        vocab_size=mc.vocab_size, hidden_size=mc.d_model, intermediate_size=mc.d_ff,
        num_hidden_layers=mc.n_layer, num_attention_heads=mc.n_head, num_key_value_heads=mc.n_kv_head,
        head_dim=mc.head_dim, max_position_embeddings=mc.seq_len, rms_norm_eps=mc.norm_eps,
        rope_theta=mc.rope_theta, tie_word_embeddings=mc.tie_embeddings, attention_bias=False,
        hidden_act="silu", bos_token_id=None, eos_token_id=[_id("<|im_end|>"), _id("<|endoftext|>")], pad_token_id=_id("<|pad|>"), torch_dtype="bfloat16",
    )
    hf = Qwen3ForCausalLM(hf_cfg)
    sd = {}
    src = model.state_dict()
    sd["model.embed_tokens.weight"] = src["embed_tokens.weight"]
    sd["model.norm.weight"] = src["norm.weight"]
    for i in range(mc.n_layer):
        p, q = f"layers.{i}.", f"model.layers.{i}."
        for a, b in (("self_attn.q_proj", "self_attn.q_proj"), ("self_attn.k_proj", "self_attn.k_proj"),
                     ("self_attn.v_proj", "self_attn.v_proj"), ("self_attn.o_proj", "self_attn.o_proj"),
                     ("self_attn.q_norm", "self_attn.q_norm"), ("self_attn.k_norm", "self_attn.k_norm"),
                     ("mlp.gate_proj", "mlp.gate_proj"), ("mlp.up_proj", "mlp.up_proj"), ("mlp.down_proj", "mlp.down_proj"),
                     ("input_layernorm", "input_layernorm"), ("post_attention_layernorm", "post_attention_layernorm")):
            sd[q + b + ".weight"] = src[p + a + ".weight"]
    if not mc.tie_embeddings:
        sd["lm_head.weight"] = src["lm_head.weight"]
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]
    assert not missing and not unexpected, (missing, unexpected)

    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(tokenizer_path),
                                  eos_token="<|endoftext|>", pad_token="<|pad|>")
    os.makedirs(out_dir, exist_ok=True)
    hf.to(torch.bfloat16).save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    # chat template and sampling defaults so `apply_chat_template` and `generate` work out of the box
    tc_path = os.path.join(out_dir, "tokenizer_config.json")
    tc = json.load(open(tc_path))
    tc["chat_template"] = ("{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + (m['content'] | trim) + '<|im_end|>\\n' }}{% endfor %}"
                           "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
    # eos_token stays <|endoftext|>: lm_eval uses it as the context prefix for log-likelihood tasks, and
    # <|im_end|> as prefix drops TurBLiMP by 10 points (2026-08-29). Chat generation stops on both ids via generation_config.
    tc["eos_token"] = "<|endoftext|>"
    tc["tokenizer_class"] = "PreTrainedTokenizerFast"  # transformers 5 writes TokenizersBackend, which llama.cpp and older loaders reject
    json.dump(tc, open(tc_path, "w"), indent=2, ensure_ascii=False)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    gen = {"eos_token_id": [im_end, tok.convert_tokens_to_ids("<|endoftext|>")], "pad_token_id": tok.convert_tokens_to_ids("<|pad|>"),
           "do_sample": True, "temperature": 0.3, "top_p": 0.9, "top_k": 40, "repetition_penalty": 1.0, "max_new_tokens": 700,
           # a 151M model falls into loops on open ended text (a poem line 28 times, "bir varmış bir yokmuş" 40
           # times). Blocking repeated 12-grams removes that (worst case loopiness 0.87 -> 0.02) and costs
           # nothing on arithmetic, whose repeated patterns are far shorter. Measured 2026-08-31.
           "no_repeat_ngram_size": 12}
    json.dump(gen, open(os.path.join(out_dir, "generation_config.json"), "w"), indent=2)
    # transformers 5 writes rope_theta only inside "rope_parameters"; transformers 4.x does not read that block and
    # silently falls back to the Qwen3 default of 10000 (found 2026-08-30 on 4.57.6: 0.21 nats mean logit error).
    # Write the legacy top-level key too so every loader gets the trained value.
    cfg_path = os.path.join(out_dir, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["rope_theta"] = float(mc.rope_theta)
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    if mc.softcap > 0:
        # ship the soft-capping model class with the weights (trust_remote_code)
        import shutil
        src_py = os.path.join(os.path.dirname(__file__), "modeling_ufakzeka.py")
        shutil.copy(src_py, os.path.join(out_dir, "modeling_ufakzeka.py"))
        cfg_path = os.path.join(out_dir, "config.json")
        cfg = json.load(open(cfg_path))
        cfg["model_type"] = "ufakzeka"
        cfg["final_logit_softcapping"] = mc.softcap
        cfg["architectures"] = ["UfakzekaForCausalLM"]
        cfg["auto_map"] = {"AutoConfig": "modeling_ufakzeka.UfakzekaConfig",
                           "AutoModelForCausalLM": "modeling_ufakzeka.UfakzekaForCausalLM"}
        json.dump(cfg, open(cfg_path, "w"), indent=2)
    # with softcap 0 the export is a stock Qwen3ForCausalLM: llama.cpp, vLLM, ONNX and transformers.js load it without custom code

    # parity check on a short input, loading the folder the way users will
    from transformers import AutoModelForCausalLM
    reloaded = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=mc.softcap > 0, dtype=torch.float32)
    ids = torch.tensor([tok.encode("Türkiye'nin başkenti Ankara'dır ve")])
    with torch.no_grad():
        ours, _ = model(ids)
        theirs = reloaded(input_ids=ids).logits
    diff = (ours - theirs).abs().max().item()
    meta = {"step": ck.get("step"), "tokens": ck.get("tokens"), "max_abs_logit_diff_vs_hf": diff}
    json.dump(meta, open(os.path.join(out_dir, "export_meta.json"), "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    export(a.ckpt, a.tokenizer, a.out)
