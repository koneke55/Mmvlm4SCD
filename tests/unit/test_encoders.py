"""Unit tests for individual modality encoders."""

import torch

from mmvlm4scd.models.encoders import (ClinicalEncoder, GenomicEncoder,
                                       ImagingEncoder, TemporalEncoder)


def test_clinical_encoder_output_shape():
    enc = ClinicalEncoder(input_dim=19, embed_dim=64, dropout=0.0)
    out = enc(torch.randn(4, 19))
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_genomic_encoder_output_shape():
    enc = GenomicEncoder(input_dim=32, embed_dim=64, dropout=0.0)
    out = enc(torch.randn(4, 32))
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_imaging_encoder_output_shape():
    enc = ImagingEncoder(input_dim=64, embed_dim=64, dropout=0.0)
    out = enc(torch.randn(4, 64))
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_temporal_encoder_output_shape():
    enc = TemporalEncoder(input_dim=6, embed_dim=64, dropout=0.0)
    out = enc(torch.randn(4, 24, 6))
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_encoders_have_trainable_params_and_grads():
    """Every encoder should have >0 trainable params and produce gradients."""
    cases = [
        (ClinicalEncoder(19, 64), torch.randn(2, 19)),
        (GenomicEncoder(32, 64),  torch.randn(2, 32)),
        (ImagingEncoder(64, 64),  torch.randn(2, 64)),
        (TemporalEncoder(6, 64),  torch.randn(2, 12, 6)),
    ]
    for enc, x in cases:
        assert sum(p.numel() for p in enc.parameters() if p.requires_grad) > 0
        x = x.clone().requires_grad_(True)
        y = enc(x)
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


def test_encoders_are_batch_size_invariant():
    """Forward should work for batch sizes 1 and N."""
    enc = ClinicalEncoder(19, 64, dropout=0.0)
    enc.eval()
    a = enc(torch.randn(1, 19))
    b = enc(torch.randn(7, 19))
    assert a.shape == (1, 64)
    assert b.shape == (7, 64)
