# Mmvlm4SCD architecture

```
                +----------------+   +----------------+
clinical x_c -->| ClinicalEnc.   |   | GenomicEnc.    |<-- genomic x_g
                +-------+--------+   +--------+-------+
                        |                     |
                        v                     v
                +-------+--------+   +--------+-------+
imaging x_i  -->| ImagingEnc.    |   | TemporalEnc.   |<-- temporal x_t
                +-------+--------+   +--------+-------+
                        |                     |
                        v                     v
                +------------------------------+
                |  Fusion (attention | cross   |
                |          | late)             |
                +-------------+----------------+
                              |
                              v
                  +-----------+-----------+
                  | Severity head (3-way) |
                  | Survival head (Cox r) |
                  +-----------------------+
```

Encoders project each modality into a shared `embed_dim`-vector. Fusion
returns a single fused vector. Two heads predict the ordinal severity
class and a continuous survival risk score. Training jointly minimises
cross-entropy + Cox partial likelihood:

```
L = alpha * CE(severity) + beta * NegPartialLik(risk, time, event).
```

See `src/mmvlm4scd/models/multimodal_model.py` and
`src/mmvlm4scd/training/losses.py`.
