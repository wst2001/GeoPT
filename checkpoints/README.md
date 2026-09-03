# Checkpoint Naming

Current checkpoint names use:

`<task>_<source>_transolver8.pt`

where `source` is either `geopt_pretrained` or `no_pretrain`.

| File | Meaning |
| --- | --- |
| `GeoPT_8layers.pt` | Original GeoPT 8-layer pretraining checkpoint used for finetuning. |
| `aircraft_geopt_pretrained_transolver8.pt` | AirCraft downstream model finetuned from GeoPT. |
| `carcrash_geopt_pretrained_transolver8.pt` | Car Crash downstream model finetuned from GeoPT. |
| `dtchull_geopt_pretrained_transolver8.pt` | DTC Hull downstream model finetuned from GeoPT. |
| `aircraft_no_pretrain_transolver8.pt` | AirCraft baseline trained without loading GeoPT pretraining. |
| `carcrash_no_pretrain_transolver8.pt` | Car Crash baseline trained without loading GeoPT pretraining. |
| `dtchull_no_pretrain_transolver8.pt` | DTC Hull baseline trained without loading GeoPT pretraining. |
