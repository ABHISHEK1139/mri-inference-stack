"""Track 4 — GAN architectures.

Re-exports every public name so ``from models.gan import build_v2_generator``
keeps working, while the implementations live in focused modules:

===========================  ==========================================
:mod:`models.gan.layers`     Shared building blocks: conditional batch
                             norm, residual blocks, self-attention
:mod:`models.gan.v2`         ResNet generator + projection discriminator
:mod:`models.gan.training`   EMA weight tracking and the WGAN-GP penalty
:mod:`models.gan.legacy`     Original DCGAN / cGAN / StyleGAN / baseline
                             builders, still selectable via ``--gan_type``
===========================  ==========================================
"""

from models.gan.layers import (
    ConditionalBatchNorm,
    DiscResBlock,
    GenResBlock,
    SelfAttention,
)
from models.gan.legacy import (
    build_baseline_discriminator,
    build_baseline_generator,
    build_conditional_discriminator,
    build_conditional_gan,
    build_conditional_generator,
    build_discriminator,
    build_gan,
    build_generator,
    build_stylegan_generator,
)
from models.gan.training import EMAGenerator, gradient_penalty
from models.gan.v2 import (
    ProjectionDiscriminator,
    ResNetGenerator,
    build_v2_discriminator,
    build_v2_generator,
)

__all__ = [
    # Shared building blocks
    "ConditionalBatchNorm",
    "DiscResBlock",
    "GenResBlock",
    "SelfAttention",
    # v2 (WGAN-GP + projection discriminator)
    "EMAGenerator",
    "ProjectionDiscriminator",
    "ResNetGenerator",
    "build_v2_discriminator",
    "build_v2_generator",
    "gradient_penalty",
    # Legacy builders, still selectable with --gan_type
    "build_baseline_discriminator",
    "build_baseline_generator",
    "build_conditional_discriminator",
    "build_conditional_gan",
    "build_conditional_generator",
    "build_discriminator",
    "build_gan",
    "build_generator",
    "build_stylegan_generator",
]
