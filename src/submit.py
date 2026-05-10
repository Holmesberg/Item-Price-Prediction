"""Write the final competition-format CSV artifact.

`python -m src.submit` is the stable entrypoint for the cleaned submission
repo. It only regenerates the best confirmed public-LB file locally:

  submissions/cand_lasso_scale1p05_shift0p08/submission.csv
"""

from __future__ import annotations

from .lb_calibration import generate


def main() -> None:
    generate(scale=1.05, shift=0.08)


if __name__ == "__main__":
    main()
