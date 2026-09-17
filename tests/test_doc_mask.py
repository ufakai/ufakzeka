"""Document masking for packed SFT sequences. Run: .venv/bin/python -m pytest tests/ -q
On CPU flex_attention runs its unfused reference path, which is enough to check the mask."""

import torch
import torch.nn.functional as F

from ufakzeka.train.model import ModelConfig, Transformer, document_block_mask

CFG = ModelConfig(vocab_size=64, n_layer=2, d_model=64, n_head=4, n_kv_head=2, d_ff=128, seq_len=64)
EOT = 63


def _packed(seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, EOT, (2, 64), generator=g)
    idx[0, 20] = EOT; idx[0, 45] = EOT  # row 0: three documents
    idx[1, 30] = EOT                     # row 1: two documents
    return idx


def test_documents_are_isolated():
    torch.manual_seed(0)
    model = Transformer(CFG).eval()
    for blk in model.layers:  # non-zero residual branches so attention actually matters
        torch.nn.init.normal_(blk.self_attn.o_proj.weight, std=0.05)
        torch.nn.init.normal_(blk.mlp.down_proj.weight, std=0.05)
    idx = _packed()
    with torch.no_grad():
        base, _ = model(idx, block_mask=document_block_mask(idx, EOT))
        changed = idx.clone(); changed[0, 5] = (changed[0, 5] + 1) % EOT  # token inside document 1 of row 0
        out, _ = model(changed, block_mask=document_block_mask(changed, EOT))
        plain, _ = model(changed)  # causal only, for contrast
    # document 1 (positions 5..20) may change, documents 2 and 3 (21..63) must not
    assert (out[0, 5:21] - base[0, 5:21]).abs().max() > 1e-4
    assert torch.allclose(out[0, 21:], base[0, 21:], atol=1e-5)
    assert torch.allclose(out[1], base[1], atol=1e-5)
    # without the mask the later documents do see the change
    assert (plain[0, 21:] - base[0, 21:]).abs().max() > 1e-4


def test_block_mask_matches_dense_reference():
    torch.manual_seed(1)
    idx = _packed(1)
    B, H, KV, T, D = 2, 4, 2, 64, 16
    q, k, v = torch.randn(B, H, T, D), torch.randn(B, KV, T, D), torch.randn(B, KV, T, D)
    from torch.nn.attention.flex_attention import flex_attention
    y = flex_attention(q, k, v, block_mask=document_block_mask(idx, EOT), enable_gqa=True)
    is_eot = (idx == EOT).long(); doc = torch.cumsum(is_eot, 1) - is_eot
    dense = (doc[:, :, None] == doc[:, None, :]) & torch.tril(torch.ones(T, T, dtype=torch.bool))
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=dense[:, None], enable_gqa=True)
    assert torch.allclose(y, ref, atol=1e-5)


def test_eot_belongs_to_its_document():
    idx = _packed()
    is_eot = (idx == EOT).long(); doc = torch.cumsum(is_eot, 1) - is_eot
    assert doc[0, 20] == 0 and doc[0, 21] == 1 and doc[0, 45] == 1 and doc[0, 46] == 2
