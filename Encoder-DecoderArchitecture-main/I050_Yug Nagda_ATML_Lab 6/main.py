from __future__ import annotations

import logging
import sys

from pipeline import run_pipeline


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        run_pipeline()
    except Exception as e:
        logging.getLogger(__name__).error("Pipeline failed: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

