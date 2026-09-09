"""Generate Geolocate instances from the prepared evidence manifest.

Instances are pooled from the reviewed evidence dataset. Each scene category
becomes one experiment holding exactly ``INSTANCES_PER_CATEGORY`` instances,
ordered by ``location_id``. Official results use the first
``OFFICIAL_INSTANCES_PER_CATEGORY`` instances of each experiment; the rest
serve as development/replacement material.
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from clemcore.clemgame import GameInstanceGenerator


DEFAULT_SEED = 1
INSTANCES_PER_CATEGORY = 10
OFFICIAL_INSTANCES_PER_CATEGORY = 5
DEFAULT_MANIFEST = "resources/dataset_manifest_final"


class GeolocateGameInstanceGenerator(GameInstanceGenerator):
    """Create stable four-view instances without exposing source metadata."""

    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed: int, **kwargs):
        lang = kwargs.get("lang", "en")
        manifest_name = kwargs.get("manifest", DEFAULT_MANIFEST)
        allow_unreviewed = bool(kwargs.get("allow_unreviewed", False))
        manifest = self.load_json(manifest_name)
        locations = manifest.get("locations", [])

        if not locations:
            raise RuntimeError(
                "No prepared Geolocate locations were found. Run "
                "geolocate/scripts/prepare_kartaview_evidence_dataset.py "
                "first to prepare the evidence dataset."
            )

        experiment_name = manifest.get("experiment_name", "kartaview_evidence_4view")
        dataset_id = manifest.get("dataset_id", "unknown")

        review_path = Path(__file__).resolve().parent / f"resources/review_{dataset_id}.json"
        if review_path.exists() and not allow_unreviewed:
            review = json.loads(review_path.read_text(encoding="utf-8"))
            decisions = review.get("locations", [])
            statuses = {
                item.get("location_id"): item.get("status")
                for item in decisions
            }
            location_ids = {item["location_id"] for item in locations}
            if set(statuses) != location_ids:
                raise RuntimeError(
                    "Review decisions do not exactly match the manifest locations."
                )
            unresolved = {
                location_id: status
                for location_id, status in statuses.items()
                if status != "keep"
            }
            if unresolved:
                counts = Counter(unresolved.values())
                raise RuntimeError(
                    "Dataset publication is blocked by manual review: "
                    f"{dict(sorted(counts.items()))}. Resolve every location to "
                    "'keep', or use --allow-unreviewed only for a deliberate smoke test."
                )

        by_category = defaultdict(list)
        for location in locations:
            category = location.get("scene_category")
            if not category:
                raise ValueError(
                    f"Location {location.get('location_id')} has no scene_category."
                )
            by_category[category].append(location)

        for category in sorted(by_category):
            ordered = sorted(
                by_category[category],
                key=lambda item: str(item.get("location_id", "")),
            )
            selected = ordered[:INSTANCES_PER_CATEGORY]

            if len(selected) < INSTANCES_PER_CATEGORY:
                raise RuntimeError(
                    f"Scene category '{category}' has only {len(selected)} "
                    f"locations; need {INSTANCES_PER_CATEGORY}."
                )

            experiment = self.add_experiment(category)
            experiment["lang"] = lang
            experiment["views_per_location"] = 4
            experiment["source"] = "KartaView"
            experiment["dataset_id"] = dataset_id
            experiment["experiment_name"] = experiment_name
            experiment["official_instance_count"] = OFFICIAL_INSTANCES_PER_CATEGORY

            for game_id, location in enumerate(selected, start=1):
                image_paths = list(location["image_paths"])

                if len(image_paths) != 4:
                    raise ValueError(
                        f"Location {location.get('location_id')} does not have four views."
                    )

                game_instance = self.add_game_instance(experiment, game_id)
                game_instance.update({
                    "image_paths": image_paths,
                    "target_latitude": float(location["latitude"]),
                    "target_longitude": float(location["longitude"]),
                    "target_country": location["country"],
                    "target_country_code": location.get("country_code"),
                })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Geolocate game instances.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="Manifest file relative to the geolocate directory, "
                             "without the .json suffix.")
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Generate development instances before all review decisions are keep.",
    )
    args = parser.parse_args()
    GeolocateGameInstanceGenerator().generate(
        seed=args.seed,
        lang="en",
        manifest=args.manifest,
        allow_unreviewed=args.allow_unreviewed,
    )
