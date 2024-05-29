# HyperWave

> [!NOTE]
> Currently in the design stage, naming and APIs all subject to change!

We're attempting to build a next-generation photonic simulation engine. We want it to be 
- **simple to use**: 
  no more bothering with transients, monitors, ring-down, etc... just solve Maxwell's equations at one or more specific wavelengths.
- **produce verifiable results**:
  return an *error term* to validate the solution can be trusted, regardless of what method was used to produce the result.
- **really, really fast**:
  state-of-the-art throughput so that no one is "just waiting for my simulation to finish".

We're currently just at the design stage: if you really care about solving Maxwell's equations, please join [our discord channel](https://discord.gg/CfzDRjeX)!

## Main API

```python
import hyperwave as hw

# Obtain the solution fields and error terms at the specified wavelengths.
field, err = hw.solve(
  wavelengths=(start, stop, num),  # Simulate at equally spaced frequencies.
  permittivity=epsilon,  # `(3, xx, yy, zz)` array.
  conductivity=sigma,  # `(3, xx, yy, zz)` array.
  source=src,  # `(3, xx, yy, zz)` array.
  error_thresh=1e-6,  # Target simulation error to achieve.
  max_fdtd_steps=10_000,  # Maximum number of FDTD step updates.
)
```