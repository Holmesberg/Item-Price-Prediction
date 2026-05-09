"""Write the final minimal submission candidate.

This wrapper exists for the familiar `python -m src.submit` entrypoint. The
actual implementation lives in `src.final_lb_candidates` so there is only one
submission path to maintain.
"""

from __future__ import annotations

from .final_lb_candidates import main


if __name__ == "__main__":
    main()
