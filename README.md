# Kineva

Pretrained encoder for representing surgical instrument kinematics.

This is the backbone model and its weights, as used in the accompanying manuscript. Pretraining, fine-tuning and evaluation code are not included.

Python 3.10+ and PyTorch 2.1+.

```
git clone https://github.com/rukun-dou/kineva-v1.git
cd kineva
pip install -r requirements.txt
```

The weights are 86 MB and tracked in the repository, so a clone is all you need.

## Use

```python
import torch
from kineva import Kineva

model = Kineva.from_pretrained("kineva_weights.pth")

x = torch.randn(1, 1200, 67)                 # (batch, timesteps, channels)
t = torch.arange(1200).float() * 0.2         # seconds

embedding = model.encode(x, t)               # (1, 384)
tokens = model.encode_patches(x, t)          # (1, 75, 384)
```

`encode` returns one vector per recording. `encode_patches` returns one token per 16 timesteps in temporal order.

## License

MIT. See `LICENSE`.

## Citation

Please cite the Kineva manuscript.

```bibtex
@misc{dou2026kineva_weights,
  title   = {Foundation Models Enable Precise Representation of Surgical Technique Through1 Self-Supervised Learning of Instrument Kinematics},
  author  = {Dou, Rukun and Uthamacumaran, Abicumaran and Ballestero, Matheus andcHaddad, Helena and Ben Bornia, Khouloud and Cattaneo, Sofia andcDahmen, Jeanne and Rinaldo, Mike and Lam, Kalista and Kang, Karman and Giglio, Bianca and Gueziri, Houssem-Eddine and Hooshiar, Amir and Del Maestro, Rolando F.},
  year    = {2026},
  version = {1.0.0}
}
```
