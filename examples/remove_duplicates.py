import argparse
import logging
from pathlib import Path

from imagededup.methods import PHash

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def remove_duplicates(input_dir: str, num_workers: int, dry_run: bool) -> None:
    image_dir = Path(input_dir).resolve()

    phasher = PHash()
    duplicates = phasher.find_duplicates(
        image_dir=image_dir,
        max_distance_threshold=8,
        outfile="duplicates.json",
        num_enc_workers=num_workers,
        num_dist_workers=num_workers,
    )

    if not dry_run:
        for _, dups in duplicates.items():
            for dup in dups:
                if Path(image_dir, dup).exists():
                    Path(image_dir, dup).unlink()

    logger.info(f"Find {len(duplicates)} images containing duplicates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove duplicates")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to the folder with images"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run  - do not delete images",
    )
    args = parser.parse_args()

    remove_duplicates(
        input_dir=args.input_dir, num_workers=args.num_workers, dry_run=args.dry_run
    )
