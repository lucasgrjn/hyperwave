# Project Goal 

We want to solve Maxwell's equations for optics in general, and photonic integrated circuits in particular. Specifically, we are looking to solve the time-harmonic wave equation for the electric field as given by

$$
\nabla \times \nabla \times E - \omega^2 \epsilon E = i \omega J.
$$

Critically, we want a solver that is
- **fast**: 
  - solves large structures in seconds
  - scales to 100s of nodes
  - doesn't break the piggybank! 
- **simple**:
  - easy to install and use
  - compatible with other tools
  - best-in-class documentation
- **open**:
  - no license servers
  - source code freely available
  - works on a variety of platforms

In other words, we want to allow photonic engineers to harness *supercomputer-level compute* in a way that will allow them to revolutionize how photonic devices are designed, manufactured, and utilized.

# Proposed Design

## Overview

```
import hyperwave as hw
field = hw.solve(omega, epsilon, source)
```

This is what we want to make possible. 

## Design choice #1: frequency-domain solutions

Although we use a time-domain method (the finite-difference time-domain, or FDTD, method) under the hood, we choose to ultimately solve in the frequency domain in order to provide:

1. *an explicit measure of convergence*: We want users to be able to independently evaluate the error in the solution field. Giving them access to the entire time-harmonic electric field will allow them to do so by directly computing an error term against the finite-difference operator.
1. *a non-leaky abstraction*: Users have access to the entirety of the electric field (and, by extension the magnetic field as well) which allows them to process the solution in any arbitrary way, and completely removed from the solution process. In contrast, time-domain solvers must embed post-processing operations within the time-stepping computation itself.
1. *a simplified interface*: No need for complicated dispersion relations and absorption characteristics. Instead, we can simply solve for the full complex permittivity at each frequency of interest indepently. Also simplifies implementation of perfectly-matched layer (PML) boundary conditions.

## Design choice #2: limit to planar devices

We deliberately limit the extent of the simulation volume along the *z*-axis in order to decrease the amount of data transfer needed per simulation step, resulting in a much faster solution. We believe this is the best choice because most practical photonic devices are, in fact, planar, and because we still allow the in-plane dimension (the *x*- and *y*-axes) to be large.

## Design choice #3: implement on AI stack

We choose to take full advantage of the tremendous investment in AI-specific hardware, software, and ecosystem development by

1. *exclusively implementing on NVIDIA GPUs*, the *de facto* leading AI-acceleration hardware platform;
2. *tightly integrating with JAX*, which allows our entire workflow to be indistinguishable from an AI-related computation; and
3. *leveraging compatibility with other AI-related tools* so that we don't reinvent technologies that are already available for data science, large-scale inference, quick prototyping, etc...

# Technical Risks

These are questions that we need the answers to... sooner rather than later!

## Numerical methods

### Is accelerated convergence possible?

While we don't absolutely need improved convergence, it would be really, really nice. I am really hoping there is some sort of artificial dispersion we can use in the simulation to suppress excitation that is not at the frequency of interest. Or something like an accelerated stationary-point method.

However, if this is not possible, we will simply plow ahead and go with the (relatively bad) convergence rate associated with the FDTD method.

### Is a simplified PML implementation possible?

It would be nice if a simplified PML implementation were possible, specifically one that did not require any auxiliary values (only complex permittivity or complex grid spacings).

If this is not possible we will fall-back on using adiabatic absorption conditions.

### What will be the effect of reduced precision?

It will be important to leverage data formats such as *bfloat16* to halve our storage and data transfer costs.

## Performance-related

### How should we design memory and data transfer?

Important to stay simple here. We can introduce our systolic scheme later, but suffice it to say that within that scheme, we will want to stack things along z. We will also probably want to allow for the source to be defined along only a plane or subvolume.

### Can we actually design a performant computation kernel?

It seems we will want to have the maximum number of warps (16) per SM, which will put a limit on the number of registers that each thread will have access to (128). Along with all the other limitations of the memory pipeline, it remains to be seen exactly how performant things can actually turn out in the end.

# De-risking Plan

1. Construct a "skeleton package"
2. Determine if accelerated convrgence is possible
3. De-risk other issues related to numerical methods
4. Build a kernel that works
5. Build a kernel that is performant