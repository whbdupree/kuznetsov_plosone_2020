This script produces simulations of "healthy BG" activity (figure 2 and figure 3 as well as figure 4 and parts of figure 6). This code is functional, but it will be more user friendly in a future iteration.

I am using the `jax` module as a numpy accelerator. I have only tested jax version 0.1.65 and jaxlib version 0.1.45. Other requirements are `numpy` and `matlotlib`.

Run the script with no arguments: `python plosone.py`. The first figure should appears in less than 10 seconds on my machine. You will need to close each figure for the next to appear. 