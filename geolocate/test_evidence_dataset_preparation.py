"""Unit tests for evidence-bearing KartaView preparation."""

import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image


REPOSITORY_DIR = Path(__file__).resolve().parent.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from geolocate.scripts.prepare_kartaview_evidence_dataset import (
    CoordinateCountryResolver,
    SCENE_CATEGORIES,
    TELEMETRY_CROP_FRACTION,
    _element_scene_category,
    _available_scene_quotas,
    _candidate_pool_ready,
    _assign_location_ids,
    _probe_configuration,
    _request_kartaview_photos,
    _review_plan,
    _sanitize_and_measure_image,
    _scene_quotas,
    _sequence_rejection_reasons,
    prepare_evidence_dataset,
    select_evidence_candidates,
)


class EvidenceDatasetPreparationTestCase(unittest.TestCase):
    def test_osm_elements_map_to_documented_scene_categories(self):
        self.assertEqual(_element_scene_category({"tags": {
            "amenity": "fuel", "name": "Example Fuel",
        }}), "roadside_commerce")
        self.assertEqual(_element_scene_category({"tags": {
            "shop": "supermarket", "name": "Example Market",
        }}), "roadside_commerce")
        self.assertEqual(_element_scene_category({"tags": {
            "highway": "motorway_junction", "ref": "12",
        }}), "transport_nodes")
        self.assertEqual(_element_scene_category({"tags": {
            "railway": "station", "name": "Example Station",
        }}), "transport_nodes")
        self.assertEqual(_element_scene_category({"tags": {
            "highway": "bus_stop", "name": "Example Stop",
        }}), "transport_nodes")
        self.assertEqual(_element_scene_category({"tags": {
            "amenity": "post_office", "name": "Example Post",
        }}), "civic_public")
        self.assertEqual(_element_scene_category({"tags": {
            "amenity": "place_of_worship", "name": "Example Church",
        }}), "civic_public")
        self.assertIsNone(_element_scene_category({"tags": {
            "tourism": "attraction", "name": "Famous landmark",
        }}))
        self.assertIsNone(_element_scene_category({"tags": {
            "amenity": "fuel",
        }}))

    def test_scene_quotas_are_balanced(self):
        self.assertEqual(_scene_quotas(30), {
            "roadside_commerce": 10,
            "transport_nodes": 10,
            "civic_public": 10,
        })

    def test_selection_ignores_country_uniqueness(self):
        candidates = []
        for category in SCENE_CATEGORIES:
            for index in range(10):
                candidates.append({
                    "scene_category": category,
                    "distance_from_feature_km": 0.1,
                    "country": "Repeated",
                    "country_code": "XX",
                    "sequence_id": f"{category}-{index}",
                    "photo_id": f"{category}-{index}",
                })
        selected = select_evidence_candidates(candidates, 30, 7)
        self.assertEqual(len(selected), 30)
        self.assertEqual({item["country_code"] for item in selected}, {"XX"})

    def test_selection_uses_near_balanced_quotas_when_one_category_is_short(self):
        candidates = []
        counts = {
            "roadside_commerce": 17,
            "transport_nodes": 12,
            "civic_public": 9,
        }
        for category, count in counts.items():
            for index in range(count):
                candidates.append({
                    "scene_category": category,
                    "distance_from_feature_km": 0.1,
                    "country": f"Country {index}",
                    "country_code": f"C{index}",
                    "sequence_id": f"{category}-{index}",
                    "photo_id": f"{category}-{index}",
                })

        self.assertEqual(_available_scene_quotas(candidates, 30), {
            "roadside_commerce": 11,
            "transport_nodes": 10,
            "civic_public": 9,
        })
        selected = select_evidence_candidates(candidates, 30, 7)
        selected_counts = {
            category: sum(item["scene_category"] == category for item in selected)
            for category in SCENE_CATEGORIES
        }
        self.assertEqual(selected_counts, {
            "roadside_commerce": 11,
            "transport_nodes": 10,
            "civic_public": 9,
        })

    def test_selection_prefers_geographic_alternatives_to_dominant_country(self):
        candidates = []
        for category in SCENE_CATEGORIES:
            for index in range(6):
                candidates.append({
                    "scene_category": category,
                    "distance_from_feature_km": 0.1,
                    "country": "Dominant",
                    "country_code": "XX",
                    "sequence_id": f"{category}-dominant-{index}",
                    "photo_id": f"{category}-dominant-{index}",
                })
            for index in range(3):
                candidates.append({
                    "scene_category": category,
                    "distance_from_feature_km": 0.1,
                    "country": f"Alternative {index}",
                    "country_code": f"A{index}",
                    "sequence_id": f"{category}-alternative-{index}",
                    "photo_id": f"{category}-alternative-{index}",
                })

        selected = select_evidence_candidates(candidates, 9, 7)
        self.assertNotIn("XX", {item["country_code"] for item in selected})

    def test_candidate_pool_must_be_balanced_before_early_stop(self):
        candidates = [
            {"scene_category": "roadside_commerce"}
            for _ in range(30)
        ]
        self.assertFalse(_candidate_pool_ready(candidates, 30))

    def test_kartaview_rate_limit_retries_the_same_probe(self):
        limited = Mock(status_code=429, headers={"Retry-After": "0"})
        success = Mock(status_code=200, headers={})
        success.raise_for_status.return_value = None
        success.json.return_value = {"result": {"data": [{"id": "photo"}]}}
        session = Mock()
        session.get.side_effect = [limited, success]

        with patch("geolocate.scripts.prepare_kartaview_evidence_dataset.time.sleep"):
            photos = _request_kartaview_photos(session, {}, 1, 2)

        self.assertEqual(photos, [{"id": "photo"}])
        self.assertEqual(session.get.call_count, 2)

    def test_probe_checkpoint_signature_tracks_category_query(self):
        config = _probe_configuration(30, 20, 8, 7, None)
        self.assertEqual(config["scene_categories"], list(SCENE_CATEGORIES))
        self.assertTrue(config["query_signature"])

    def test_review_plan_preserves_keep_and_pending_and_excludes_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "manifest": root / "manifest.json",
                "review_json": root / "review.json",
            }
            paths["manifest"].write_text(json.dumps({"locations": [
                {
                    "location_id": "location_001",
                    "source_photo_id": "keep",
                    "source_sequence_id": "sequence-keep",
                    "country": "A",
                    "scene_category": "civic_public",
                },
                {
                    "location_id": "location_002",
                    "source_photo_id": "pending",
                    "source_sequence_id": "sequence-pending",
                    "country": "B",
                    "scene_category": "transport_nodes",
                },
                {
                    "location_id": "location_003",
                    "source_photo_id": "reject",
                    "source_sequence_id": "sequence-reject",
                    "country": "C",
                    "scene_category": "roadside_commerce",
                },
            ]}))
            paths["review_json"].write_text(json.dumps({"locations": [
                {"location_id": "location_001", "status": "keep", "notes": ""},
                {"location_id": "location_002", "status": "pending", "notes": "check"},
                {"location_id": "location_003", "status": "reject", "notes": "bad"},
            ]}))

            required, exclusions, _, previous_manifest = _review_plan(paths, True)

        self.assertEqual(required, {"keep", "pending"})
        self.assertEqual(exclusions[0]["photo_id"], "reject")
        self.assertEqual(exclusions[0]["kind"], "manual_review")
        selected = _assign_location_ids([
            {"photo_id": "keep", "scene_category": "civic_public"},
            {"photo_id": "pending", "scene_category": "transport_nodes"},
            {"photo_id": "replacement", "scene_category": "roadside_commerce"},
        ], previous_manifest)
        assigned = {item["photo_id"]: item["_location_id"] for item in selected}
        self.assertEqual(assigned, {
            "keep": "location_001",
            "pending": "location_002",
            "replacement": "location_003",
        })

    def test_country_resolver_uses_coordinate_cache_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "countries.json"
            cache.write_text(json.dumps({
                "dataset_id": "test",
                "entries": {
                    "47.927632,17.103542": {
                        "country": "Austria",
                        "country_code": "AT",
                    },
                },
            }))
            session = Mock()
            resolver = CoordinateCountryResolver(session, cache, "test")
            result = resolver.resolve(47.927632, 17.103542)

        self.assertEqual(result, ("Austria", "AT"))
        session.get.assert_not_called()

    def test_telemetry_band_is_cropped_before_publication(self):
        source = Image.new("RGB", (100, 100), "white")
        buffer = BytesIO()
        source.save(buffer, "JPEG")
        image, _ = _sanitize_and_measure_image(buffer.getvalue(), 1000)
        self.assertEqual(image.height, round(100 * (1 - TELEMETRY_CROP_FRACTION)))

    def test_flat_green_sequence_is_rejected(self):
        image = Image.new("RGB", (100, 100), (20, 120, 20))
        buffer = BytesIO()
        image.save(buffer, "JPEG")
        prepared, metrics = _sanitize_and_measure_image(buffer.getvalue(), 1000)
        reasons = _sequence_rejection_reasons([metrics] * 4, [prepared] * 4)
        self.assertIn("all_views_vegetation_dominated", reasons)
        self.assertIn("fewer_than_three_technically_usable_views", reasons)

    def test_preparation_writes_manifest_attribution_and_review(self):
        candidates = []
        for category_index, category in enumerate(SCENE_CATEGORIES):
            candidates.append({
                "scene_category": category,
                "distance_from_feature_km": 0.1,
                "country": f"Country {category_index}",
                "country_code": f"C{category_index}",
                "sequence_id": f"sequence-{category_index}",
                "photo_id": f"photo-{category_index}",
                "contributor_user_id": "user",
                "latitude": 1.0,
                "longitude": 2.0,
                "shot_date": None,
                "frame_span_km": 0.05,
                "photos": [{
                    "photo_id": f"photo-{category_index}-{view_index}",
                    "source_image_url": f"url-{category_index}-{view_index}",
                } for view_index in range(4)],
            })

        source = Image.new("RGB", (128, 128), "white")
        for x in range(0, 128, 8):
            for y in range(0, 128, 8):
                if (x + y) // 8 % 2:
                    for pixel_x in range(x, x + 8):
                        for pixel_y in range(y, y + 8):
                            source.putpixel((pixel_x, pixel_y), (20, 20, 20))
        image_bytes = []
        for view_index in range(4):
            variant = source.copy()
            for x in range(view_index * 16, view_index * 16 + 12):
                for y in range(24, 104):
                    variant.putpixel((x, y), (180, 40, 40))
            buffer = BytesIO()
            variant.save(buffer, "JPEG")
            image_bytes.append(buffer.getvalue())

        def download(_session, source_url):
            return image_bytes[int(source_url.rsplit("-", 1)[1])]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "manifest": root / "manifest.json",
                "attribution": root / "attribution.json",
                "exclusions": root / "exclusions.json",
                "images": root / "images_evidence_v1",
                "review_html": root / "review.html",
                "review_json": root / "review.json",
            }
            with patch(
                "geolocate.scripts.prepare_kartaview_evidence_dataset._download_image",
                side_effect=download,
            ):
                selected = prepare_evidence_dataset(
                    object(), candidates, paths, 3, 128, 7,
                    "evidence_v1", "kartaview_evidence_4view",
                )

            self.assertEqual(len(selected), 3)
            self.assertTrue(paths["manifest"].exists())
            self.assertTrue(paths["attribution"].exists())
            self.assertTrue(paths["review_html"].exists())
            self.assertTrue(paths["review_json"].exists())
            self.assertEqual(len(list(paths["images"].rglob("*.jpg"))), 12)


if __name__ == "__main__":
    unittest.main()
