"""Pure unit tests for deterministic KartaView hard-set preparation."""

import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image


REPOSITORY_DIR = Path(__file__).resolve().parent.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from geolocate.scripts.prepare_kartaview_dataset import (
    SAMPLING_BANDS,
    _dataset_paths,
    _destination_point,
    _final_band_quotas,
    _haversine_km,
    _probe_plan,
    prepare_dataset,
    select_balanced_candidates,
)


class KartaViewDatasetPreparationTestCase(unittest.TestCase):
    def test_destination_point_respects_distance(self):
        latitude, longitude = _destination_point(48.1351, 11.5820, 100.0, 90.0)
        distance = _haversine_km(48.1351, 11.5820, latitude, longitude)
        self.assertAlmostEqual(distance, 100.0, places=6)

    def test_probe_plan_is_deterministic_and_inside_annuli(self):
        seeds = [{
            "name": "Anchor",
            "latitude": 48.1351,
            "longitude": 11.5820,
            "country": "Germany",
            "country_code": "DE",
        }]
        first = _probe_plan(seeds, probes_per_seed_per_band=3, random_seed=17)
        second = _probe_plan(seeds, probes_per_seed_per_band=3, random_seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 9)

        bands = {band["name"]: band for band in SAMPLING_BANDS}
        for probe in first:
            distance = _haversine_km(
                seeds[0]["latitude"],
                seeds[0]["longitude"],
                probe["latitude"],
                probe["longitude"],
            )
            band = bands[probe["sampling_band"]]
            self.assertGreaterEqual(distance, band["minimum_km"])
            self.assertLessEqual(distance, band["maximum_km"])

    def test_hard_paths_do_not_reuse_easy_control_paths(self):
        paths = _dataset_paths("hard_v1")
        self.assertEqual(paths["images"].name, "images_hard_v1")
        self.assertEqual(paths["manifest"].name, "dataset_manifest_hard_v1.json")
        self.assertEqual(
            paths["download_exclusions"].name,
            "download_exclusions_hard_v1.json",
        )
        self.assertNotEqual(paths["images"].name, "images")

    def test_dataset_id_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            _dataset_paths("../hard")

    def test_balanced_selection_uses_documented_bands(self):
        candidates = []
        photo_id = 0
        for band_index, band in enumerate(SAMPLING_BANDS):
            for country_index in range(12):
                photo_id += 1
                candidates.append({
                    "sampling_band": band["name"],
                    "country": f"Country {band_index}-{country_index}",
                    "country_code": f"{band_index}{country_index:02d}",
                    "sequence_id": str(photo_id),
                    "photo_id": str(photo_id),
                })

        selected = select_balanced_candidates(candidates, limit=30, random_seed=9)
        counts = {
            band["name"]: sum(
                candidate["sampling_band"] == band["name"]
                for candidate in selected
            )
            for band in SAMPLING_BANDS
        }
        self.assertEqual(counts, {
            "peripheral": 12,
            "secondary_town": 9,
            "rural_interurban": 9,
        })

    def test_final_band_quotas_use_documented_allocation(self):
        self.assertEqual(_final_band_quotas(30), {
            "peripheral": 12,
            "secondary_town": 9,
            "rural_interurban": 9,
        })
        self.assertEqual(_final_band_quotas(9), {
            "peripheral": 3,
            "secondary_town": 3,
            "rural_interurban": 3,
        })

    def test_selection_rejects_candidate_too_far_from_probe(self):
        candidates = [
            {
                "sampling_band": "peripheral",
                "country": "Eligible",
                "country_code": "OK",
                "sequence_id": "1",
                "photo_id": "1",
                "distance_from_probe_km": 0.5,
            },
            {
                "sampling_band": "peripheral",
                "country": "Outlier",
                "country_code": "NO",
                "sequence_id": "2",
                "photo_id": "2",
                "distance_from_probe_km": 7.36,
            },
        ]
        selected = select_balanced_candidates(candidates, limit=1, random_seed=4)
        self.assertEqual(selected[0]["country_code"], "OK")

    def test_balanced_selection_is_reproducible(self):
        candidates = []
        for index, band in enumerate(SAMPLING_BANDS):
            for country_index in range(4):
                value = index * 10 + country_index
                candidates.append({
                    "sampling_band": band["name"],
                    "country": str(value),
                    "country_code": str(value),
                    "sequence_id": str(value),
                    "photo_id": str(value),
                })

        first = select_balanced_candidates(candidates, limit=9, random_seed=4)
        second = select_balanced_candidates(candidates, limit=9, random_seed=4)
        self.assertEqual(first, second)

    def test_selection_allows_repeated_countries_without_changing_band_mix(self):
        candidates = []
        quotas = _final_band_quotas(30)
        for band_name, quota in quotas.items():
            for index in range(quota):
                candidates.append({
                    "sampling_band": band_name,
                    "country": "Repeated country",
                    "country_code": "XX",
                    "sequence_id": f"{band_name}-{index}",
                    "photo_id": f"{band_name}-{index}",
                })

        selected = select_balanced_candidates(candidates, 30, 9)
        counts = {
            band_name: sum(
                item["sampling_band"] == band_name for item in selected
            )
            for band_name in quotas
        }
        self.assertEqual(counts, {
            "peripheral": 12,
            "secondary_town": 9,
            "rural_interurban": 9,
        })
        self.assertEqual({item["country_code"] for item in selected}, {"XX"})

    def test_download_failure_reselects_without_partial_dataset(self):
        def candidate(name: str, photo_id: str, source_prefix: str) -> dict:
            return {
                "sampling_band": "peripheral",
                "country": name,
                "country_code": name[:2].upper(),
                "sequence_id": photo_id,
                "photo_id": photo_id,
                "latitude": 1.0,
                "longitude": 2.0,
                "shot_date": None,
                "distance_from_anchor_km": 20.0,
                "distance_from_probe_km": 0.5,
                "frame_span_km": 0.05,
                "contributor_user_id": "user",
                "photos": [
                    {
                        "photo_id": f"{photo_id}-{index}",
                        "sequence_index": index,
                        "source_image_url": f"{source_prefix}-{index}",
                    }
                    for index in range(4)
                ],
            }

        bad = candidate("Unavailable", "bad", "bad-url")
        good = candidate("Replacement", "good", "good-url")
        image_buffer = BytesIO()
        Image.new("RGB", (8, 8), "blue").save(image_buffer, format="JPEG")
        image_bytes = image_buffer.getvalue()

        def download(_session, source_url):
            if source_url.startswith("bad-url"):
                raise RuntimeError("archived blob")
            return image_bytes

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "candidates": root / "candidates.json",
                "manifest": root / "manifest.json",
                "attribution": root / "attribution.json",
                "download_exclusions": root / "download_exclusions.json",
                "images": root / "images_hard_v1",
            }
            with (
                patch(
                    "geolocate.scripts.prepare_kartaview_dataset."
                    "select_balanced_candidates",
                    side_effect=[[bad], [good]],
                ),
                patch(
                    "geolocate.scripts.prepare_kartaview_dataset._download_image",
                    side_effect=download,
                ),
            ):
                selected = prepare_dataset(
                    object(),
                    [bad, good],
                    limit=1,
                    view_size=32,
                    paths=paths,
                    dataset_id="hard_v1",
                    experiment_name="kartaview_hard_4view",
                    random_seed=4,
                )

            self.assertEqual([item["photo_id"] for item in selected], ["good"])
            manifest = json.loads(paths["manifest"].read_text())
            self.assertEqual(manifest["locations"][0]["country"], "Replacement")
            exclusions = manifest["selection"]["download_exclusions"]
            self.assertEqual(exclusions[0]["photo_id"], "bad")
            persisted = json.loads(paths["download_exclusions"].read_text())
            self.assertEqual(
                persisted["download_exclusions"][0]["photo_id"],
                "bad",
            )
            self.assertEqual(len(list(paths["images"].rglob("*.jpg"))), 4)

    def test_download_exclusion_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "manifest": root / "manifest.json",
                "attribution": root / "attribution.json",
                "download_exclusions": root / "download_exclusions.json",
                "images": root / "images_hard_v1",
            }
            paths["download_exclusions"].write_text(json.dumps({
                "dataset_id": "hard_v1",
                "download_exclusions": [{"photo_id": "bad"}],
            }))
            good = {
                "sampling_band": "peripheral",
                "country": "Replacement",
                "country_code": "OK",
                "sequence_id": "good",
                "photo_id": "good",
                "latitude": 1.0,
                "longitude": 2.0,
                "shot_date": None,
                "distance_from_anchor_km": 20.0,
                "distance_from_probe_km": 0.5,
                "frame_span_km": 0.05,
                "contributor_user_id": "user",
                "photos": [{
                    "photo_id": f"good-{index}",
                    "sequence_index": index,
                    "source_image_url": f"good-{index}",
                } for index in range(4)],
            }
            bad = dict(good, photo_id="bad", country="Unavailable")
            image_buffer = BytesIO()
            Image.new("RGB", (8, 8), "blue").save(image_buffer, format="JPEG")

            with patch(
                "geolocate.scripts.prepare_kartaview_dataset._download_image",
                return_value=image_buffer.getvalue(),
            ) as download:
                selected = prepare_dataset(
                    object(), [bad, good], 1, 32, paths,
                    "hard_v1", "kartaview_hard_4view", 4,
                )

            self.assertEqual(selected[0]["photo_id"], "good")
            self.assertEqual(download.call_count, 4)


if __name__ == "__main__":
    unittest.main()
