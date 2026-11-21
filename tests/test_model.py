"""Our decoder against the Hugging Face implementation, on small random models
(CPU, fp32): prefill logits, then several decoding steps through the cache."""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from gpuvintage.engine import GraphDecoder, prefill_all, span_for, spans_for  # noqa: E402
from gpuvintage.model import Model  # noqa: E402


def _tiny(tmp_path, arch: str, tie: bool):
    torch.manual_seed(0)
    common = dict(vocab_size=320, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                  rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=tie)
    if arch == "qwen3":
        cfg = transformers.Qwen3Config(head_dim=24, **common)  # head_dim != hidden/heads
        hf = transformers.Qwen3ForCausalLM(cfg)
    else:
        cfg = transformers.Qwen2Config(**common)
        hf = transformers.Qwen2ForCausalLM(cfg)
    # random norms, so that a missing or misplaced norm shows
    with torch.no_grad():
        for n, p in hf.named_parameters():
            if "norm" in n:
                p.uniform_(0.5, 1.5)
            if n.endswith("bias"):
                p.normal_(0.0, 0.1)
    hf.eval()
    d = tmp_path / arch
    hf.save_pretrained(d, safe_serialization=True)
    return hf, d


@pytest.mark.parametrize("arch,tie", [("qwen3", True), ("qwen3", False), ("qwen2", True)])
def test_prefill_and_decode_match_hf(tmp_path, arch, tie):
    hf, d = _tiny(tmp_path, arch, tie)
    m = Model.from_pretrained(d, device="cpu", dtype=torch.float32)
    assert m.num_parameters() == sum(p.numel() for p in hf.parameters())
    B, t, steps = 3, 11, 6
    g = torch.Generator().manual_seed(1)
    prompts = torch.randint(0, 320, (B, t), generator=g)
    cache = m.new_cache(B, t + steps)

    ours = m.prefill(prompts, cache)
    with torch.no_grad():
        ref = hf(prompts, use_cache=True)
    torch.testing.assert_close(ours, ref.logits[:, -1], rtol=1e-4, atol=1e-4)

    # greedy decoding, compared step by step with HF's own cache
    past = ref.past_key_values
    nxt = torch.argmax(ref.logits[:, -1], -1)
    tok = nxt.clone()
    pos = torch.tensor([t])
    for s in range(steps):
        lg = m.decode_step(tok, pos, cache, span=t + steps)
        with torch.no_grad():
            r = hf(nxt[:, None], past_key_values=past, use_cache=True)
        past = r.past_key_values
        torch.testing.assert_close(lg, r.logits[:, -1], rtol=1e-4, atol=1e-4)
        nxt = torch.argmax(r.logits[:, -1], -1)
        tok = torch.argmax(lg, -1)
        assert torch.equal(tok, nxt)
        pos += 1


def test_eager_decoder_and_chunked_prefill_cpu(tmp_path):
    hf, d = _tiny(tmp_path, "qwen3", True)
    m = Model.from_pretrained(d, device="cpu", dtype=torch.float32)
    B, t, steps = 5, 9, 7
    prompts = torch.randint(0, 320, (B, t), generator=torch.Generator().manual_seed(2))
    c1 = m.new_cache(B, t + steps)
    c2 = m.new_cache(B, t + steps)
    a = prefill_all(m, prompts, c1, max_tokens=10_000)
    b = prefill_all(m, prompts, c2, max_tokens=2 * t)  # three micro-batches
    assert torch.equal(a, b)
    torch.testing.assert_close(c1, c2)
    dec = GraphDecoder(m, c1, t, t + steps, bucket=4, use_graphs=False)
    dec.reset(a)
    dec.run(steps)
    assert int(dec.pos) == t + steps
    out = hf.generate(prompts, max_new_tokens=steps + 1, do_sample=False, min_new_tokens=steps + 1)
    # the decoder's last token is the (steps+1)-th generated token
    assert torch.equal(dec.tok, out[:, t + steps])


def test_spans():
    assert spans_for(512, 768, 64) == [576, 640, 704, 768]
    assert spans_for(500, 768, 64) == [512, 576, 640, 704, 768]
    assert spans_for(10, 13, 4) == [12, 13]
    assert span_for(511, [512, 576]) == 512
    assert span_for(512, [512, 576]) == 576
    with pytest.raises(ValueError):
        span_for(576, [512, 576])
