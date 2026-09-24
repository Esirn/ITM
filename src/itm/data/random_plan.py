"""Order-independent sampling noise for same-backbone comparisons.

Every draw is addressed by case, sampling seed and diffusion step. A model's
extra random draws cannot shift another branch's stream. Training seeds are not
part of this interface. Models must still integrate this into their sampler.
"""
import hashlib
import json

import numpy as np


VERSION = "itm-pcg64-addressed-noise-v1"


def sampling_noise(shape, *, backbone_id, case_id, sample_seed, step):
    if not backbone_id or not case_id or not isinstance(sample_seed, int) or not isinstance(step, int) or step < -1:
        raise ValueError("backbone, case, integer sampling seed and step >= -1 required")
    if not shape or any(not isinstance(d, int) or d < 1 for d in shape):
        raise ValueError("positive integer noise shape required")
    address = {"version": VERSION, "backbone_id": backbone_id, "case_id": case_id,
               "sample_seed": sample_seed, "step": step, "shape": list(shape)}
    digest = hashlib.sha256(json.dumps(address, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    generator = np.random.Generator(np.random.PCG64(int(digest, 16)))
    noise = generator.standard_normal(shape).astype(np.float32)
    return noise, {**address, "address_sha256": digest, "numpy_version": np.__version__,
                   "noise_sha256": hashlib.sha256(noise.tobytes(order="C")).hexdigest(),
                   "dtype": "float32", "initial_noise_step": -1}
