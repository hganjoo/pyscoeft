# PySCo-EFT v2
### By Himanish Ganjoo
### LUX, Observatoire de Paris
### March 2026

PySCo-EFT is based on PySCo by Michel-Andrès Breton (https://github.com/mianbreton/pysco)

In addition to standard PySCo, it implements the cubic screening model in the Effective Field Theory of Dark Energy formalism.
Based on Cusin et. al. (2018) -- https://arxiv.org/abs/1712.02782

## Parameters:

The examples files demonstrate the usage of the EFT theory parameters.

theory: 'eft'

- eftlin: Turn linearised EFT on or off
- alphaB0: braiding parameter at z = 0
- alphaM0: Planck mass run rate at z = 0
- scaling: the time variation of the alphaB and alphaM, proportional to DE density ('de') or scale factor ('a')
- nb: the exponent for the time variation of alphaB
- nm: the exponent for the time variation of alphaM
- Npre_FAS: the number of pre-cycles for the EFT multigrid
- Npost_FAS: the number of post-cycles for the EFT multigrid

