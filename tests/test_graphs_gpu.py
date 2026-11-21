"""CUDA-graph decoding against eager decoding (needs a GPU)."""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from gpuvintage.engine import GraphDecoder, prefill_all  # noqa: E402
from gpuvintage.model import Model  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_graph_decode_matches_eager(tmp_path):
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=320, hidden_size=64, intermediate_size=96,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=32, tie_word_embeddings=True)
    transformers.Qwen3ForCausalLM(cfg).save_pretrained(tmp_path, safe_serialization=True)
    m = Model.from_pretrained(tmp_path, device="cuda", dtype=torch.float32)
    B, t, steps = 4, 20, 40
    prompts = torch.randint(0, 320, (B, t), generator=torch.Generator().manual_seed(3)).cuda()
    toks = {}
    for graphs in (False, True):
        cache = m.new_cache(B, t + steps)
        first = prefill_all(m, prompts, cache)
        dec = GraphDecoder(m, cache, t, t + steps, bucket=8, use_graphs=graphs)
        for _ in range(2):  # a second pass must restart cleanly
            dec.reset(first)
            dec.run(steps)
        toks[graphs] = dec.tok.clone()
        assert int(dec.pos) == t + steps
    assert torch.equal(toks[False], toks[True])
