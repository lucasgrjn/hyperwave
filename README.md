# HyperWave

We're building an AI-powered photonic simulation engine that is simple, fast, free, and scalable. Please join us!

## Vision

We want to provide the basic building block needed for computational photonics... the ability to solve Maxwell's equations.

Unfortunately, although solving the equations for light is hard enough as is, today's solutions do not make things any easier. Photonic engineers are currently left to choose between tools which are either outdated, expensive, or closed-source. The end result being that photonic simulation and design capabilities have stagnated for decades.

*Why can't computational photonics blossom in the same way AI has?* We realized that while AI had TensorFlow, PyTorch, and a whole foundational technology layer that was performant, open, and free, no such thing existed for photonics to this day! And instead of being discouraged, we also realized that we could quickly catch up by simply latching on to the incredible innovation produced by the AI boom: after all, why re-invent the wheel right?

And that's what we want to invite you to build with us. A technology stack that revolutionizes how photonics is designed and built by embedding itself in the AI hardware, software, and tooling ecosystem.

## Feature Set

In general, we choose to follow a minimalistic, lean solution which prioritizes performance. For this reason, we attempt to cut out any non-critical features, and limit the scope of the project as much as possible while still being applicable to the vast majority of photonic simulation needs.

First, being both inspired and powered by the AI ecosystem, we have decided to
- remain open-source (MIT license),
- exclusively target NVIDIA GPUs, and
- to embed natively integrate into the AI software stack ([JAX](https://github.com/google/jax)).

In terms of simulation capabilities, our underlying [FDTD](https://en.wikipedia.org/wiki/Finite-difference_time-domain_method) implementation
- supports arbitrary refractive index and conductivity spatial distribution,
- does not support frequency/wavelength material dispersion,
- uses adiabatically absorbing boundary conditions in place of perfectly-matched layers, and
- allows for snapshotting simulation subvolumes at specific timesteps, 
- does not allow for running output computations within the simulation loop.

## Current Status

HyperWave is currently in the initial development phase, if you have any design or implementation insights, feature requests for your use case, or general questions we would love to chat! Please [open an issue](https://github.com/spinsphotonics/hyperwave/issues/new) or join our Slack!

## Who We Are

We're the brains behind [fdtd-z](https://github.com/spinsphotonics/fdtdz)... HyperWave is our second crack at this problem!