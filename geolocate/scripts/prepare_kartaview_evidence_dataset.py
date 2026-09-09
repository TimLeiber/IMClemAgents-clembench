"""Prepare an evidence-bearing KartaView dataset for Geolocate.

This pipeline deliberately separates discovery from benchmark publication:

1. discover OpenStreetMap features likely to produce usable visual clues;
2. find compact four-frame KartaView sequences near those features;
3. download, sanitize, validate, and publish a versioned image set;
4. generate a review page for the mandatory human quality audit.

The OSM feature name and identifier are retained only in the private candidate
audit file. They are never copied into game instances or the public manifest.
"""

import argparse
import hashlib
import html
import json
import math
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from .prepare_kartaview_dataset import (
        LICENSE_NAME,
        LICENSE_URL,
        PHOTO_ENDPOINT,
        REPOSITORY_DIR,
        RESOURCES_DIR,
        SEEDS_PATH,
        _download_image,
        _expanded_sequence_burst,
        _haversine_km,
        _read_json,
        _write_json,
    )
except ImportError:
    from prepare_kartaview_dataset import (
        LICENSE_NAME,
        LICENSE_URL,
        PHOTO_ENDPOINT,
        REPOSITORY_DIR,
        RESOURCES_DIR,
        SEEDS_PATH,
        _download_image,
        _expanded_sequence_burst,
        _haversine_km,
        _read_json,
        _write_json,
    )


DEFAULT_DATASET_ID = "final"
DEFAULT_EXPERIMENT_NAME = "kartaview_evidence_4view"
DEFAULT_LOCATION_LIMIT = 30
DEFAULT_CANDIDATE_POOL_SIZE = 120
DEFAULT_RANDOM_SEED = 20260902
DEFAULT_FEATURE_RADIUS_KM = 30
# Features must sit outside this annulus floor around the anchor's city-centre
# coordinate. A model that only recognizes the anchor city and aims at its
# centre scores inverse_quality(floor distance); at 20 km that shortcut scores
# ~83, safely below the ~91 "good" threshold, so reaching a good score
# requires actually reading the local evidence.
MIN_FEATURE_DISTANCE_FROM_ANCHOR_KM = 20.0
DEFAULT_FEATURES_PER_SEED_CATEGORY = 8
DEFAULT_KARTAVIEW_RADIUS_METERS = 400
MAX_TARGET_DISTANCE_FROM_FEATURE_KM = 0.35
TELEMETRY_CROP_FRACTION = 0.12
PROBE_CHECKPOINT_SCHEMA_VERSION = 2
CANDIDATE_CHECKPOINT_SCHEMA_VERSION = 2
KARTAVIEW_API_ATTEMPTS = 5
KARTAVIEW_API_BACKOFF_SECONDS = (5, 15, 30, 60)
NOMINATIM_REVERSE_ENDPOINT = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_MINIMUM_INTERVAL_SECONDS = 1.05
NOMINATIM_ATTEMPTS = 3
NOMINATIM_BACKOFF_SECONDS = (5, 15)
SCENE_CATEGORIES = (
    "roadside_commerce",
    "transport_nodes",
    "civic_public",
)
ROADSIDE_COMMERCE_AMENITIES = ("fuel", "bank", "pharmacy", "fast_food")
ROADSIDE_COMMERCE_SHOPS = ("supermarket",)
CIVIC_PUBLIC_KINDS = ("post_office", "townhall", "place_of_worship", "school")
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
OVERPASS_ATTEMPTS_PER_TILE = 3
OVERPASS_ATTEMPT_BACKOFF_SECONDS = (5, 15)
# Maximum sampling-square span (2 x radius) queried in a single Overpass
# request. The earlier 30 km-radius runs used one ``around:`` query per anchor
# and completed quickly, so anything within this span stays a single query;
# larger radii are split into tiles of roughly this size.
SINGLE_QUERY_SPAN_KM = 60.0
DEFAULT_MAX_ANCHORS = None


def _preparation_session() -> requests.Session:
    """Create a session whose retries also cover nested sequence-page calls."""
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount(PHOTO_ENDPOINT, adapter)
    session.headers["User-Agent"] = "clembench-geolocate-evidence-preparation/1.0"
    return session


def _paths(dataset_id: str) -> dict[str, Path]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", dataset_id):
        raise ValueError(
            "dataset_id must contain only lowercase letters, digits, underscores, and hyphens"
        )
    return {
        "probes": RESOURCES_DIR / f"osm_evidence_probes_{dataset_id}.json",
        "candidates": RESOURCES_DIR / f"kartaview_candidates_{dataset_id}.json",
        "manifest": RESOURCES_DIR / f"dataset_manifest_{dataset_id}.json",
        "attribution": RESOURCES_DIR / f"attribution_{dataset_id}.json",
        "exclusions": RESOURCES_DIR / f"selection_exclusions_{dataset_id}.json",
        "countries": RESOURCES_DIR / f"coordinate_countries_{dataset_id}.json",
        "images": RESOURCES_DIR / f"images_{dataset_id}",
        "review_html": RESOURCES_DIR / f"review_{dataset_id}.html",
        "review_json": RESOURCES_DIR / f"review_{dataset_id}.json",
    }


class CoordinateCountryResolver:
    """Resolve and cache the country containing each target coordinate."""

    def __init__(self,
                 session: requests.Session,
                 cache_path: Path,
                 dataset_id: str):
        self.session = session
        self.cache_path = cache_path
        self.dataset_id = dataset_id
        document = (
            _read_json(cache_path)
            if cache_path.exists()
            else {"dataset_id": dataset_id, "entries": {}}
        )
        if document.get("dataset_id") != dataset_id:
            raise RuntimeError(
                f"Country cache {cache_path} belongs to dataset "
                f"{document.get('dataset_id')!r}, not {dataset_id!r}."
            )
        self.entries = document.get("entries", {})
        self.last_request_at = 0.0
        seeds = _read_json(SEEDS_PATH)
        self.country_names = {
            str(seed["country_code"]).upper(): seed["country"]
            for seed in seeds
        }

    @staticmethod
    def _key(latitude: float, longitude: float) -> str:
        return f"{float(latitude):.6f},{float(longitude):.6f}"

    def _write(self) -> None:
        _write_json(self.cache_path, {
            "dataset_id": self.dataset_id,
            "source": "OpenStreetMap Nominatim reverse geocoding",
            "entries": self.entries,
        })

    def resolve(self, latitude: float, longitude: float) -> tuple[str, str]:
        key = self._key(latitude, longitude)
        cached = self.entries.get(key)
        if cached:
            return cached["country"], cached["country_code"]

        last_error = None
        document = None
        for attempt in range(1, NOMINATIM_ATTEMPTS + 1):
            wait = NOMINATIM_MINIMUM_INTERVAL_SECONDS - (
                time.monotonic() - self.last_request_at
            )
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.session.get(
                    NOMINATIM_REVERSE_ENDPOINT,
                    params={
                        "format": "jsonv2",
                        "lat": float(latitude),
                        "lon": float(longitude),
                        "zoom": 10,
                        "addressdetails": 1,
                    },
                    timeout=(20, 45),
                )
                self.last_request_at = time.monotonic()
                response.raise_for_status()
                document = response.json()
                break
            except (requests.RequestException, ValueError) as error:
                last_error = error
                if attempt < NOMINATIM_ATTEMPTS:
                    time.sleep(NOMINATIM_BACKOFF_SECONDS[attempt - 1])
        if document is None:
            raise RuntimeError(
                f"Country lookup failed for coordinate {key}: {last_error}"
            ) from last_error
        address = document.get("address") or {}
        country_code = str(address.get("country_code") or "").upper()
        if not country_code:
            raise RuntimeError(
                f"Could not resolve a country for coordinate {key}."
            )
        country = self.country_names.get(
            country_code,
            str(address.get("country") or country_code),
        )
        self.entries[key] = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "country": country,
            "country_code": country_code,
        }
        self._write()
        return country, country_code


def _request_overpass(session: requests.Session, query: str) -> dict:
    errors = []
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = session.post(
                endpoint,
                data={"data": query},
                timeout=(30, 180),
            )
            response.raise_for_status()
            document = response.json()
            if not isinstance(document, dict):
                raise ValueError("Overpass response is not a JSON object")
            remark = document.get("remark")
            if remark:
                raise ValueError(f"Overpass returned an incomplete result: {remark}")
            if not isinstance(document.get("elements"), list):
                raise RuntimeError("Overpass response has no elements list")
            return document
        except (requests.RequestException, ValueError) as error:
            errors.append(f"{endpoint}: {error}")
    raise RuntimeError("All Overpass endpoints failed: " + " | ".join(errors))


def _overpass_tile_query(bbox: str) -> str:
    """Query named evidence features inside one bounding box tile.

    A single 100 km ``around:`` union repeatedly exceeded public Overpass
    server limits (HTTP 504) on dense anchors. The annulus is therefore
    scanned as small bbox tiles, each roughly the size of the 30 km queries
    that previously succeeded; the annulus shape is enforced client-side
    afterwards.
    """
    commerce_amenity_regex = "|".join(ROADSIDE_COMMERCE_AMENITIES)
    commerce_shop_regex = "|".join(ROADSIDE_COMMERCE_SHOPS)
    civic_regex = "|".join(CIVIC_PUBLIC_KINDS)
    return f"""[out:json][timeout:120];
(
  nwr({bbox})["amenity"~"{commerce_amenity_regex}"]["name"];
  nwr({bbox})["shop"~"{commerce_shop_regex}"]["name"];
  nwr({bbox})["railway"="station"]["name"];
  node({bbox})["highway"="motorway_junction"]["ref"];
  node({bbox})["highway"="bus_stop"]["name"];
  node({bbox})["railway"="tram_stop"]["name"];
  nwr({bbox})["amenity"~"{civic_regex}"]["name"];
);
out tags center;"""


def _annulus_tiles(seed: dict,
                   radius_km: float) -> list[str]:
    """Split the anchor's outer sampling square into Overpass bbox strings.

    A sampling square whose full span fits within :data:`SINGLE_QUERY_SPAN_KM`
    is queried as one tile (the fast behaviour of the earlier 30 km runs);
    larger spans are subdivided into tiles of roughly that size so no single
    Overpass request exceeds what the public instances will compute.
    """
    latitude = float(seed["latitude"])
    longitude = float(seed["longitude"])
    delta_latitude = radius_km / 110.574
    delta_longitude = radius_km / (111.320 * math.cos(math.radians(latitude)))
    span_km = 2.0 * radius_km
    grid = max(1, math.ceil(span_km / SINGLE_QUERY_SPAN_KM))
    tiles = []
    for row in range(grid):
        for column in range(grid):
            south = latitude - delta_latitude + 2 * delta_latitude * row / grid
            north = latitude - delta_latitude + 2 * delta_latitude * (row + 1) / grid
            west = longitude - delta_longitude + 2 * delta_longitude * column / grid
            east = longitude - delta_longitude + 2 * delta_longitude * (column + 1) / grid
            tiles.append(f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}")
    return tiles


def _element_coordinates(element: dict) -> tuple[float, float] | None:
    if "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])
    center = element.get("center") or {}
    if "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    return None


def _element_scene_category(element: dict) -> str | None:
    tags = element.get("tags") or {}
    named = bool(tags.get("name"))
    if named and (
        tags.get("amenity") in ROADSIDE_COMMERCE_AMENITIES
        or tags.get("shop") in ROADSIDE_COMMERCE_SHOPS
    ):
        return "roadside_commerce"
    if (
        tags.get("highway") == "motorway_junction" and tags.get("ref")
    ) or (
        named and (
            tags.get("railway") in {"station", "tram_stop"}
            or tags.get("highway") == "bus_stop"
        )
    ):
        return "transport_nodes"
    if named and tags.get("amenity") in CIVIC_PUBLIC_KINDS:
        return "civic_public"
    return None


def _stable_rank(value: str, random_seed: int) -> str:
    return hashlib.sha256(f"{random_seed}:{value}".encode("utf-8")).hexdigest()


def _probe_configuration(radius_km: float,
                         min_distance_km: float,
                         features_per_seed_category: int,
                         random_seed: int,
                         max_anchors: int | None) -> dict:
    """Return every setting that changes the discovered probe population."""
    query_signature = hashlib.sha256(
        _overpass_tile_query("BBOX").encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": PROBE_CHECKPOINT_SCHEMA_VERSION,
        "radius_km": radius_km,
        "min_distance_from_anchor_km": min_distance_km,
        "features_per_seed_category": features_per_seed_category,
        "random_seed": random_seed,
        "max_anchors": max_anchors,
        "scene_categories": list(SCENE_CATEGORIES),
        "query_signature": query_signature,
    }


def discover_osm_evidence_probes(session: requests.Session,
                                 output_path: Path,
                                 radius_km: float,
                                 features_per_seed_category: int,
                                 random_seed: int,
                                 dataset_id: str,
                                 min_distance_km: float = MIN_FEATURE_DISTANCE_FROM_ANCHOR_KM,
                                 max_anchors: int | None = DEFAULT_MAX_ANCHORS) -> list[dict]:
    """Discover clue-bearing OSM features without selecting famous landmarks.

    ``max_anchors`` limits discovery to a deterministic subset of the seed
    list so the Overpass load stays bounded; ``None`` uses every seed.
    """
    seeds = _read_json(SEEDS_PATH)
    if max_anchors is not None and max_anchors < len(seeds):
        ranked = sorted(
            seeds,
            key=lambda item: _stable_rank(f"anchor:{item['name']}", random_seed),
        )
        seeds = ranked[:max_anchors]
    if output_path.exists():
        checkpoint = _read_json(output_path)
        expected = _probe_configuration(
            radius_km,
            min_distance_km,
            features_per_seed_category,
            random_seed,
            max_anchors,
        )
        actual = {key: checkpoint.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"Existing probe checkpoint uses {actual}, not {expected}. "
                "Use a new dataset ID or remove that checkpoint explicitly."
            )
        probes = checkpoint.get("probes", [])
        failures = checkpoint.get("failures", [])
        anchor_statistics = checkpoint.get("anchor_statistics", {})
        completed_anchors = set(checkpoint.get("completed_anchors", []))
    else:
        probes = []
        failures = []
        anchor_statistics = {}
        completed_anchors = set()
    seen_elements = {
        f"{probe['osm_element_type']}:{probe['osm_element_id']}"
        for probe in probes
    }

    def write_checkpoint(complete: bool) -> None:
        ordered = sorted(probes, key=lambda item: _stable_rank(
            f"{item['osm_element_type']}:{item['osm_element_id']}",
            random_seed,
        ))
        configuration = _probe_configuration(
            radius_km,
            min_distance_km,
            features_per_seed_category,
            random_seed,
            max_anchors,
        )
        _write_json(output_path, {
            "dataset_id": dataset_id,
            "source": "OpenStreetMap via Overpass API",
            "license": "ODbL 1.0",
            "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
            "strategy": "named_non_landmark_evidence_features",
            **configuration,
            "complete": complete,
            "completed_anchors": sorted(completed_anchors),
            "anchor_statistics": anchor_statistics,
            "failures": failures,
            "probes": ordered,
        })

    for seed_index, seed in enumerate(seeds, start=1):
        if seed["name"] in completed_anchors:
            continue
        elements: list[dict] = []
        tile_failed = None
        for tile in _annulus_tiles(seed, radius_km):
            document = None
            for attempt in range(1, OVERPASS_ATTEMPTS_PER_TILE + 1):
                try:
                    document = _request_overpass(session, _overpass_tile_query(tile))
                    break
                except RuntimeError as error:
                    if attempt < OVERPASS_ATTEMPTS_PER_TILE:
                        backoff = OVERPASS_ATTEMPT_BACKOFF_SECONDS[min(
                            attempt - 1, len(OVERPASS_ATTEMPT_BACKOFF_SECONDS) - 1,
                        )]
                        print(
                            f"OSM discovery tile attempt {attempt} failed for "
                            f"{seed['name']}; retrying in {backoff}s",
                            flush=True,
                        )
                        time.sleep(backoff)
                    else:
                        tile_failed = error
            if tile_failed is not None:
                break
            if document is not None:
                elements.extend(document.get("elements", []))
        if tile_failed is not None:
            failures.append({"seed": seed["name"], "reason": str(tile_failed)})
            anchor_statistics[seed["name"]] = {
                "status": "request_failed",
                "reason": str(tile_failed),
            }
            print(
                f"OSM feature discovery failed for {seed['name']}: {tile_failed}",
                flush=True,
            )
            write_checkpoint(False)
            continue

        if not elements:
            reason = "Overpass returned zero features for every tile"
            failures.append({"seed": seed["name"], "reason": reason})
            anchor_statistics[seed["name"]] = {
                "status": "empty_response",
                "tiles": len(_annulus_tiles(seed, radius_km)),
            }
            print(
                f"OSM feature discovery produced no data for {seed['name']}; "
                "leaving the anchor incomplete so it can be retried",
                flush=True,
            )
            write_checkpoint(False)
            continue

        by_category = defaultdict(list)
        for element in elements:
            category = _element_scene_category(element)
            coordinates = _element_coordinates(element)
            element_key = f"{element.get('type')}:{element.get('id')}"
            if category is None or coordinates is None or element_key in seen_elements:
                continue

            latitude, longitude = coordinates
            distance_from_seed = _haversine_km(
                float(seed["latitude"]),
                float(seed["longitude"]),
                latitude,
                longitude,
            )
            # Features inside the annulus floor around the anchor would let a
            # model score well by merely recognizing the anchor city and aiming
            # at its centre; the floor keeps that shortcut below the good-score
            # threshold. Tile corners can extend beyond the sampling radius.
            if distance_from_seed < min_distance_km or distance_from_seed > radius_km:
                continue

            tags = element.get("tags") or {}
            by_category[category].append({
                "scene_category": category,
                "latitude": latitude,
                "longitude": longitude,
                "anchor_name": seed["name"],
                "anchor_country": seed["country"],
                "anchor_country_code": seed["country_code"],
                "distance_from_anchor_km": distance_from_seed,
                "osm_element_type": element.get("type"),
                "osm_element_id": str(element.get("id")),
                "osm_feature_name": tags.get("name") or tags.get("ref"),
                "osm_feature_kind": (
                    tags.get("amenity")
                    or tags.get("shop")
                    or tags.get("railway")
                    or tags.get("highway")
                ),
            })

        before_count = len(probes)
        eligible_by_category = {
            category: len(by_category[category])
            for category in SCENE_CATEGORIES
        }
        for category in SCENE_CATEGORIES:
            ranked = sorted(
                by_category[category],
                key=lambda item: _stable_rank(
                    f"{item['osm_element_type']}:{item['osm_element_id']}",
                    random_seed,
                ),
            )
            for probe in ranked[:features_per_seed_category]:
                key = f"{probe['osm_element_type']}:{probe['osm_element_id']}"
                if key not in seen_elements:
                    seen_elements.add(key)
                    probes.append(probe)

        completed_anchors.add(seed["name"])
        anchor_statistics[seed["name"]] = {
            "status": "completed",
            "features_returned": len(elements),
            "eligible_by_category": eligible_by_category,
            "probes_added": len(probes) - before_count,
        }
        write_checkpoint(False)
        added_count = sum(len(items) for items in by_category.values())
        print(
            f"OSM anchors {seed_index}/{len(seeds)}: {seed['name']} "
            f"({len(probes)} probes total; {len(elements)} features returned, "
            f"{added_count} eligible before per-category limits)",
            flush=True,
        )
        time.sleep(0.25)

    probes.sort(key=lambda item: _stable_rank(
        f"{item['osm_element_type']}:{item['osm_element_id']}",
        random_seed,
    ))
    write_checkpoint(len(completed_anchors) == len(seeds))
    print(f"Prepared {len(probes)} OSM evidence probes in {output_path}")
    return probes


def audit_kartaview_evidence(session: requests.Session,
                             probes: list[dict],
                             output_path: Path,
                             radius_meters: int,
                             target_count: int | None,
                             dataset_id: str) -> list[dict]:
    """Find compact KartaView sequences close to evidence-bearing features."""
    seeds = _read_json(SEEDS_PATH)
    country_names = {seed["country_code"]: seed["country"] for seed in seeds}
    probe_signature = hashlib.sha256(
        json.dumps([
            {
                "key": f"{probe['osm_element_type']}:{probe['osm_element_id']}",
                "category": probe["scene_category"],
                "latitude": probe["latitude"],
                "longitude": probe["longitude"],
            }
            for probe in sorted(
                probes,
                key=lambda item: (
                    str(item["osm_element_type"]),
                    str(item["osm_element_id"]),
                ),
            )
        ], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_config = {
        "schema_version": CANDIDATE_CHECKPOINT_SCHEMA_VERSION,
        "maximum_target_distance_from_feature_km": (
            MAX_TARGET_DISTANCE_FROM_FEATURE_KM
        ),
        "kartaview_query_radius_meters": radius_meters,
        "scene_categories": list(SCENE_CATEGORIES),
        "probe_population_signature": probe_signature,
    }
    if output_path.exists():
        checkpoint = _read_json(output_path)
        actual_config = {
            key: checkpoint.get("sampling", {}).get(key)
            for key in expected_config
        }
        if actual_config != expected_config:
            raise RuntimeError(
                f"Existing candidate checkpoint uses {actual_config}, not "
                f"{expected_config}. Use a new dataset ID or remove it explicitly."
            )
        candidates = checkpoint.get("candidates", [])
        processed_probe_keys = set(checkpoint.get("processed_probe_keys", []))
        rejections = Counter(checkpoint.get("sampling", {}).get("rejections", {}))
    else:
        candidates = []
        processed_probe_keys = set()
        rejections = Counter()
    seen_sequences = {candidate["sequence_id"] for candidate in candidates}

    def probe_key(probe: dict) -> str:
        return f"{probe['osm_element_type']}:{probe['osm_element_id']}"

    def write_checkpoint(complete: bool) -> None:
        _write_json(output_path, {
            "dataset_id": dataset_id,
            "complete": complete,
            "processed_probe_keys": sorted(processed_probe_keys),
            "sampling": {
                "strategy": "osm_evidence_feature_probes",
                **expected_config,
                "scene_categories": list(SCENE_CATEGORIES),
                "probe_attempts": len(processed_probe_keys),
                "rejections": dict(sorted(rejections.items())),
            },
            "candidates": candidates,
        })

    if _candidate_pool_ready(candidates, target_count):
        return candidates

    for attempt_number, probe in enumerate(probes, start=1):
        key = probe_key(probe)
        if key in processed_probe_keys:
            continue
        try:
            photos = _request_kartaview_photos(
                session,
                {
                    "lat": probe["latitude"],
                    "lng": probe["longitude"],
                    "zoomLevel": 18,
                    "radius": radius_meters,
                    "join": "sequence",
                    "orderBy": "id",
                    "orderDirection": "desc",
                },
                attempt_number,
                len(probes),
            )
        except (requests.RequestException, ValueError) as error:
            rejections["kartaview_api_error"] += 1
            print(
                f"KartaView probe {attempt_number}/{len(probes)} failed: {error}",
                flush=True,
            )
            write_checkpoint(False)
            continue

        burst = _expanded_sequence_burst(
            session,
            photos,
            {"latitude": probe["latitude"], "longitude": probe["longitude"]},
        )
        if burst is None:
            rejections["no_four_frame_sequence"] += 1
            processed_probe_keys.add(key)
            if len(processed_probe_keys) % 10 == 0:
                write_checkpoint(False)
            continue

        photo = burst[1]
        sequence = photo.get("sequence") or {}
        sequence_id = str(sequence.get("id", ""))
        if not sequence_id or sequence_id in seen_sequences:
            rejections["duplicate_sequence"] += 1
            processed_probe_keys.add(key)
            continue

        latitude = float(photo["lat"])
        longitude = float(photo["lng"])
        feature_distance = _haversine_km(
            float(probe["latitude"]),
            float(probe["longitude"]),
            latitude,
            longitude,
        )
        if feature_distance > MAX_TARGET_DISTANCE_FROM_FEATURE_KM:
            rejections["too_far_from_evidence_feature"] += 1
            processed_probe_keys.add(key)
            continue

        country_code = str(sequence.get("countryCode") or probe["anchor_country_code"])
        country = country_names.get(country_code, country_code)
        frame_span = max(
            _haversine_km(
                latitude,
                longitude,
                float(item["lat"]),
                float(item["lng"]),
            )
            for item in burst
        )
        candidate = {
            **probe,
            "country": country,
            "country_code": country_code,
            "distance_from_feature_km": feature_distance,
            "frame_span_km": frame_span,
            "photo_id": str(photo["id"]),
            "sequence_id": sequence_id,
            "contributor_user_id": str(sequence.get("userId", "")),
            "latitude": latitude,
            "longitude": longitude,
            "heading": float(photo.get("heading") or 0),
            "shot_date": photo.get("shotDate"),
            "photos": [{
                "photo_id": str(item["id"]),
                "sequence_index": int(item["sequenceIndex"]),
                "latitude": float(item["lat"]),
                "longitude": float(item["lng"]),
                "source_image_url": item["fileurlProc"],
            } for item in burst],
        }
        candidates.append(candidate)
        seen_sequences.add(sequence_id)
        processed_probe_keys.add(key)
        write_checkpoint(False)
        print(
            f"evidence candidate {len(candidates)}: {country} "
            f"{probe['scene_category']} ({feature_distance:.2f} km from feature)",
            flush=True,
        )
        if _candidate_pool_ready(candidates, target_count):
            break

    complete = _candidate_pool_ready(
        candidates, target_count
    ) or len(processed_probe_keys) == len(probes)
    write_checkpoint(complete)
    category_counts = Counter(item["scene_category"] for item in candidates)
    print(
        f"Found {len(candidates)} evidence candidates in {output_path}; "
        f"categories={dict(sorted(category_counts.items()))}"
    )
    return candidates


def _retry_after_seconds(response: requests.Response,
                         attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return float(KARTAVIEW_API_BACKOFF_SECONDS[min(
        attempt - 1,
        len(KARTAVIEW_API_BACKOFF_SECONDS) - 1,
    )])


def _request_kartaview_photos(session: requests.Session,
                              params: dict,
                              probe_number: int,
                              probe_count: int) -> list[dict]:
    """Request one probe, respecting KartaView throttling before moving on."""
    last_error = None
    for attempt in range(1, KARTAVIEW_API_ATTEMPTS + 1):
        try:
            response = session.get(
                PHOTO_ENDPOINT,
                params=params,
                timeout=(20, 45),
            )
            if response.status_code == 429 and attempt < KARTAVIEW_API_ATTEMPTS:
                delay = _retry_after_seconds(response, attempt)
                print(
                    f"KartaView probe {probe_number}/{probe_count} was rate "
                    f"limited; retrying in {delay:g}s "
                    f"({attempt}/{KARTAVIEW_API_ATTEMPTS})",
                    flush=True,
                )
                time.sleep(delay)
                continue
            response.raise_for_status()
            document = response.json()
            photos = document.get("result", {}).get("data", [])
            if not isinstance(photos, list):
                raise ValueError("KartaView response has no photo list")
            return photos
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt >= KARTAVIEW_API_ATTEMPTS:
                break
            delay = float(KARTAVIEW_API_BACKOFF_SECONDS[min(
                attempt - 1,
                len(KARTAVIEW_API_BACKOFF_SECONDS) - 1,
            )])
            print(
                f"KartaView probe {probe_number}/{probe_count} failed: {error}; "
                f"retrying in {delay:g}s ({attempt}/{KARTAVIEW_API_ATTEMPTS})",
                flush=True,
            )
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("KartaView request failed without an exception")


def _candidate_pool_ready(candidates: list[dict],
                          target_count: int | None) -> bool:
    """Require a balanced pool before stopping an audit early."""
    if target_count is None or len(candidates) < target_count:
        return False
    quotas = _scene_quotas(target_count)
    counts = Counter(item.get("scene_category") for item in candidates)
    return all(counts[category] >= quota for category, quota in quotas.items())


def _scene_quotas(limit: int) -> dict[str, int]:
    base, remainder = divmod(limit, len(SCENE_CATEGORIES))
    return {
        category: base + (index < remainder)
        for index, category in enumerate(SCENE_CATEGORIES)
    }


def _available_scene_quotas(candidates: list[dict],
                            limit: int) -> dict[str, int]:
    """Balance categories as closely as the usable candidate pool permits."""
    available = Counter(
        candidate.get("scene_category")
        for candidate in candidates
        if (
            candidate.get("scene_category") in SCENE_CATEGORIES
            and float(candidate.get("distance_from_feature_km", math.inf))
            <= MAX_TARGET_DISTANCE_FROM_FEATURE_KM
        )
    )
    if sum(available.values()) < limit:
        raise RuntimeError(
            f"Evidence pool has {sum(available.values())} usable candidates; "
            f"{limit} are required. Available by category: "
            f"{dict(sorted(available.items()))}."
        )

    quotas = {category: 0 for category in SCENE_CATEGORIES}
    category_order = {category: index for index, category in enumerate(SCENE_CATEGORIES)}
    for _ in range(limit):
        choices = [
            category for category in SCENE_CATEGORIES
            if quotas[category] < available[category]
        ]
        category = min(
            choices,
            key=lambda item: (quotas[item], category_order[item]),
        )
        quotas[category] += 1
    return quotas


def _diverse_balanced_sample(candidates: list[dict],
                             quotas: dict[str, int],
                             random_seed: int) -> list[dict]:
    """Meet scene quotas while preferring underrepresented geographies."""
    selected = []
    by_category = defaultdict(list)
    for candidate in candidates:
        by_category[candidate["scene_category"]].append(candidate)

    for category, quota in quotas.items():
        choices = by_category[category]
        geography_sizes = Counter(
            str(
                candidate.get("country_code")
                or candidate.get("country")
                or candidate.get("anchor_name")
                or "unknown"
            )
            for candidate in choices
        )

        def rank(candidate: dict) -> tuple:
            geography = str(
                candidate.get("country_code")
                or candidate.get("country")
                or candidate.get("anchor_name")
                or "unknown"
            )
            stable = _stable_rank(
                f"{category}:{candidate['sequence_id']}:{candidate['photo_id']}",
                random_seed,
            )
            return geography_sizes[geography], stable

        selected.extend(sorted(choices, key=rank)[:quota])

    return selected


def select_evidence_candidates(candidates: list[dict],
                               limit: int,
                               random_seed: int,
                               required_photo_ids: set[str] | None = None) -> list[dict]:
    """Select near-balanced scenes while softly spreading geography."""
    quotas = _available_scene_quotas(candidates, limit)
    grouped = defaultdict(list)
    for candidate in candidates:
        if (
            candidate.get("scene_category") in quotas
            and float(candidate.get("distance_from_feature_km", math.inf))
            <= MAX_TARGET_DISTANCE_FROM_FEATURE_KM
        ):
            grouped[candidate["scene_category"]].append(candidate)

    required_photo_ids = {str(item) for item in (required_photo_ids or set())}
    candidates_by_photo = {
        str(candidate["photo_id"]): candidate
        for category in SCENE_CATEGORIES
        for candidate in grouped[category]
    }
    missing_required = required_photo_ids - set(candidates_by_photo)
    if missing_required:
        raise RuntimeError(
            "Reviewed keep/pending candidates are absent from the candidate "
            f"pool: {sorted(missing_required)}"
        )
    required = [candidates_by_photo[item] for item in sorted(required_photo_ids)]
    required_counts = Counter(item["scene_category"] for item in required)
    overfilled = {
        category: (required_counts[category], quotas[category])
        for category in SCENE_CATEGORIES
        if required_counts[category] > quotas[category]
    }
    if overfilled:
        raise RuntimeError(
            f"Reviewed candidates exceed the final scene quotas: {overfilled}"
        )

    remaining_quotas = {
        category: quotas[category] - required_counts[category]
        for category in SCENE_CATEGORIES
    }
    remaining = [
        candidate for candidate in candidates_by_photo.values()
        if str(candidate["photo_id"]) not in required_photo_ids
    ]
    selected = required + _diverse_balanced_sample(
        remaining,
        remaining_quotas,
        random_seed,
    )
    rng = random.Random(random_seed)
    rng.shuffle(selected)
    return selected


def _sanitize_and_measure_image(image_bytes: bytes,
                                view_size: int) -> tuple[Image.Image, dict]:
    with Image.open(BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    crop_height = max(1, round(image.height * (1.0 - TELEMETRY_CROP_FRACTION)))
    image = image.crop((0, 0, image.width, crop_height))
    image.thumbnail((view_size, view_size), Image.Resampling.LANCZOS)

    analysis = image.copy()
    analysis.thumbnail((192, 192), Image.Resampling.BILINEAR)
    pixels = np.asarray(analysis, dtype=np.float32)
    gray = pixels.mean(axis=2)
    horizontal = np.abs(np.diff(gray, axis=1))
    vertical = np.abs(np.diff(gray, axis=0))
    gradient = (horizontal.mean() + vertical.mean()) / 2.0
    edge_density = (
        (horizontal > 18).mean() + (vertical > 18).mean()
    ) / 2.0
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    vegetation = (
        (green > red * 1.08)
        & (green > blue * 1.05)
        & (green > 45)
    ).mean()
    return image, {
        "brightness": round(float(gray.mean()), 3),
        "contrast": round(float(gray.std()), 3),
        "gradient": round(float(gradient), 3),
        "edge_density": round(float(edge_density), 4),
        "vegetation_fraction": round(float(vegetation), 4),
    }


def _sequence_rejection_reasons(metrics: list[dict],
                                thumbnails: list[Image.Image]) -> list[str]:
    reasons = []
    usable = [
        metric["contrast"] >= 15
        and metric["gradient"] >= 2.0
        and 20 <= metric["brightness"] <= 240
        for metric in metrics
    ]
    if sum(usable) < 3:
        reasons.append("fewer_than_three_technically_usable_views")

    vegetation = [metric["vegetation_fraction"] for metric in metrics]
    if all(value >= 0.52 for value in vegetation):
        reasons.append("all_views_vegetation_dominated")

    arrays = []
    for image in thumbnails:
        copy = image.convert("L")
        copy.thumbnail((96, 96), Image.Resampling.BILINEAR)
        arrays.append(np.asarray(copy, dtype=np.float32))
    differences = [
        float(np.abs(left - right).mean())
        for left, right in zip(arrays, arrays[1:])
        if left.shape == right.shape
    ]
    if differences and max(differences) < 0.75:
        reasons.append("four_views_are_effectively_duplicates")
    return reasons


class EvidenceCandidateError(RuntimeError):
    def __init__(self, candidate: dict, kind: str, reason: str):
        self.candidate = candidate
        self.kind = kind
        self.reason = reason
        super().__init__(reason)


def _materialize_candidate(session: requests.Session,
                           candidate: dict,
                           location_dir: Path,
                           view_size: int,
                           image_cache: dict[str, bytes]) -> tuple[list[str], list[dict]]:
    prepared = []
    metrics = []
    for photo in candidate["photos"]:
        source_url = photo["source_image_url"]
        try:
            image_bytes = image_cache.get(source_url)
            if image_bytes is None:
                image_bytes = _download_image(session, source_url)
                image_cache[source_url] = image_bytes
            image, image_metrics = _sanitize_and_measure_image(image_bytes, view_size)
        except (RuntimeError, UnidentifiedImageError, OSError, ValueError) as error:
            raise EvidenceCandidateError(
                candidate,
                "download_or_decode_failure",
                str(error),
            ) from error
        prepared.append(image)
        metrics.append(image_metrics)

    reasons = _sequence_rejection_reasons(metrics, prepared)
    if reasons:
        raise EvidenceCandidateError(candidate, "visual_quality", ",".join(reasons))

    image_paths = []
    for view_index, image in enumerate(prepared):
        output_path = location_dir / f"view_{view_index:02d}.jpg"
        image.save(output_path, "JPEG", quality=92, optimize=True)
        image_paths.append(output_path.name)
    return image_paths, metrics


def _write_review(paths: dict[str, Path],
                  manifest: dict,
                  previous_review: dict | None = None,
                  previous_manifest: dict | None = None) -> None:
    previous_locations = {
        item["location_id"]: item
        for item in (previous_manifest or {}).get("locations", [])
    }
    previous_decisions_by_photo = {}
    for decision in (previous_review or {}).get("locations", []):
        old_location = previous_locations.get(decision.get("location_id"))
        photo_id = (
            decision.get("source_photo_id")
            or (old_location or {}).get("source_photo_id")
        )
        if photo_id:
            previous_decisions_by_photo[str(photo_id)] = {
                "status": decision.get("status", "pending"),
                "notes": decision.get("notes", ""),
            }

    review_locations = []
    cards = []
    for location in manifest["locations"]:
        decision = previous_decisions_by_photo.get(
            str(location["source_photo_id"]),
            {"status": "pending", "notes": ""},
        )
        review_location = {
            "location_id": location["location_id"],
            "source_photo_id": location["source_photo_id"],
            "status": decision["status"],
            "notes": decision["notes"],
            "scene_category": location["scene_category"],
            "country": location["country"],
        }
        review_locations.append(review_location)
        images = "".join(
            f'<img src="{html.escape(Path(path).relative_to("geolocate/resources").as_posix())}" '
            f'alt="{html.escape(location["location_id"])} view">'
            for path in location["image_paths"]
        )
        cards.append(
            f'<section><h2>{html.escape(location["location_id"])} · '
            f'{html.escape(location["country"])} · '
            f'{html.escape(location["scene_category"])}</h2>'
            f'<div class="views">{images}</div>'
            f'<pre>{html.escape(json.dumps(location["image_metrics"], indent=2))}</pre>'
            f'<p>Current decision: <strong>{html.escape(decision["status"])}</strong>'
            f' &nbsp; Notes: {html.escape(decision["notes"] or "—")}</p></section>'
        )

    review_status = (
        "complete"
        if review_locations and all(
            item["status"] in {"keep", "reject"}
            for item in review_locations
        )
        else "pending"
    )
    _write_json(paths["review_json"], {
        "dataset_id": manifest["dataset_id"],
        "review_status": review_status,
        "instructions": (
            "Keep only sequences with useful but non-trivial visual evidence. "
            "Reject landmarks, telemetry leakage, unreadable scenes, and clue-free roads."
        ),
        "locations": review_locations,
    })
    paths["review_html"].write_text(
        "<!doctype html><meta charset='utf-8'><title>Geolocate evidence review</title>"
        "<style>body{font:15px system-ui;margin:24px;background:#eee}"
        "section{background:white;padding:16px;margin:0 0 24px;border-radius:8px}"
        ".views{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}"
        "img{width:100%;height:220px;object-fit:contain;background:#111}"
        "pre{font-size:11px;white-space:pre-wrap}</style>"
        "<h1>Geolocate evidence-v1 review</h1>"
        "<p>Coordinates and OSM feature names are intentionally omitted.</p>"
        + "".join(cards),
        encoding="utf-8",
    )


def _review_plan(paths: dict[str, Path],
                 apply_review: bool) -> tuple[set[str], list[dict], dict | None, dict | None]:
    previous_manifest = (
        _read_json(paths["manifest"])
        if paths["manifest"].exists()
        else None
    )
    previous_review = (
        _read_json(paths["review_json"])
        if paths["review_json"].exists()
        else None
    )
    if not apply_review:
        return set(), [], previous_review, previous_manifest
    if previous_manifest is None or previous_review is None:
        raise RuntimeError(
            "--apply-review requires an existing manifest and review JSON."
        )

    manifest_by_id = {
        item["location_id"]: item
        for item in previous_manifest.get("locations", [])
    }
    decisions = previous_review.get("locations", [])
    decision_ids = [item.get("location_id") for item in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise RuntimeError("Review JSON contains duplicate location IDs.")
    if set(decision_ids) != set(manifest_by_id):
        raise RuntimeError(
            "Review JSON location IDs do not exactly match the current manifest."
        )

    required_photo_ids = set()
    manual_exclusions = []
    for decision in decisions:
        status = decision.get("status")
        if status not in {"keep", "reject", "pending"}:
            raise RuntimeError(
                f"Invalid review status {status!r} for "
                f"{decision.get('location_id')}."
            )
        location = manifest_by_id[decision["location_id"]]
        photo_id = str(location["source_photo_id"])
        if status in {"keep", "pending"}:
            required_photo_ids.add(photo_id)
            continue
        manual_exclusions.append({
            "photo_id": photo_id,
            "sequence_id": location["source_sequence_id"],
            "country": location["country"],
            "scene_category": location["scene_category"],
            "kind": "manual_review",
            "reason": decision.get("notes") or "Rejected during manual review",
        })
    return required_photo_ids, manual_exclusions, previous_review, previous_manifest


def _assign_location_ids(selected: list[dict],
                         previous_manifest: dict | None) -> list[dict]:
    """Keep reviewed location IDs stable and reuse rejected category slots."""
    selected = [dict(candidate) for candidate in selected]
    if previous_manifest is None:
        for index, candidate in enumerate(selected, start=1):
            candidate["_location_id"] = f"location_{index:03d}"
        return selected

    previous_locations = previous_manifest.get("locations", [])
    old_by_photo = {
        str(item["source_photo_id"]): item
        for item in previous_locations
    }
    used_ids = set()
    for candidate in selected:
        old = old_by_photo.get(str(candidate["photo_id"]))
        if old:
            candidate["_location_id"] = old["location_id"]
            used_ids.add(old["location_id"])

    open_ids = defaultdict(list)
    for old in previous_locations:
        if old["location_id"] not in used_ids:
            open_ids[old["scene_category"]].append(old["location_id"])
    for category in open_ids:
        open_ids[category].sort()

    for candidate in selected:
        if "_location_id" in candidate:
            continue
        category = candidate["scene_category"]
        if not open_ids[category]:
            raise RuntimeError(
                f"No open location slot remains for category {category!r}."
            )
        candidate["_location_id"] = open_ids[category].pop(0)
    return sorted(selected, key=lambda item: item["_location_id"])


def prepare_evidence_dataset(session: requests.Session,
                             candidates: list[dict],
                             paths: dict[str, Path],
                             limit: int,
                             view_size: int,
                             random_seed: int,
                             dataset_id: str,
                             experiment_name: str,
                             apply_review: bool = False) -> list[dict]:
    required_photo_ids, manual_exclusions, previous_review, previous_manifest = (
        _review_plan(paths, apply_review)
    )
    exclusions_document = (
        _read_json(paths["exclusions"])
        if paths["exclusions"].exists()
        else {"dataset_id": dataset_id, "exclusions": []}
    )
    exclusions = exclusions_document.get("exclusions", [])
    exclusion_ids = {str(item["photo_id"]) for item in exclusions}
    for exclusion in manual_exclusions:
        if str(exclusion["photo_id"]) not in exclusion_ids:
            exclusions.append(exclusion)
            exclusion_ids.add(str(exclusion["photo_id"]))
    if manual_exclusions:
        _write_json(paths["exclusions"], {
            "dataset_id": dataset_id,
            "exclusions": exclusions,
        })
    excluded_ids = {str(item["photo_id"]) for item in exclusions}
    image_cache = {}
    staging = paths["images"].with_name(paths["images"].name + ".staging")
    previous_by_photo = {
        str(item["source_photo_id"]): item
        for item in (previous_manifest or {}).get("locations", [])
    }
    country_resolver = (
        CoordinateCountryResolver(session, paths["countries"], dataset_id)
        if "countries" in paths
        else None
    )

    while True:
        shutil.rmtree(staging, ignore_errors=True)
        selected = _assign_location_ids(select_evidence_candidates(
            [item for item in candidates if str(item["photo_id"]) not in excluded_ids],
            limit,
            random_seed,
            required_photo_ids,
        ), previous_manifest if apply_review else None)
        staging.mkdir(parents=True)
        locations = []
        failed = None

        for candidate in selected:
            location_id = candidate["_location_id"]
            location_dir = staging / location_id
            location_dir.mkdir()
            if country_resolver is not None:
                country, country_code = country_resolver.resolve(
                    candidate["latitude"], candidate["longitude"],
                )
            else:
                country = candidate["country"]
                country_code = candidate["country_code"]
            print(
                f"Preparing {location_id} ({country}, "
                f"{candidate['scene_category']})",
                flush=True,
            )
            try:
                previous = previous_by_photo.get(str(candidate["photo_id"]))
                previous_paths = [
                    (
                        Path(path)
                        if Path(path).is_absolute()
                        else REPOSITORY_DIR / path
                    )
                    for path in (previous or {}).get("image_paths", [])
                ]
                if (
                    previous
                    and len(previous_paths) == 4
                    and all(path.exists() for path in previous_paths)
                ):
                    filenames = []
                    for source in previous_paths:
                        destination = location_dir / source.name
                        shutil.copy2(source, destination)
                        filenames.append(destination.name)
                    metrics = previous["image_metrics"]
                else:
                    filenames, metrics = _materialize_candidate(
                        session, candidate, location_dir, view_size, image_cache,
                    )
            except EvidenceCandidateError as error:
                failed = error
                break

            image_paths = [
                (
                    Path("geolocate/resources")
                    / paths["images"].name
                    / location_id
                    / filename
                ).as_posix()
                for filename in filenames
            ]
            locations.append({
                "location_id": location_id,
                "image_paths": image_paths,
                "latitude": candidate["latitude"],
                "longitude": candidate["longitude"],
                "country": country,
                "country_code": country_code,
                "scene_category": candidate["scene_category"],
                "source_photo_id": candidate["photo_id"],
                "source_sequence_id": candidate["sequence_id"],
                "shot_date": candidate["shot_date"],
                "distance_from_feature_km": candidate["distance_from_feature_km"],
                "frame_span_km": candidate["frame_span_km"],
                "image_metrics": metrics,
            })

        if failed is None:
            break
        candidate = failed.candidate
        if str(candidate["photo_id"]) in required_photo_ids:
            raise RuntimeError(
                f"Reviewed {candidate['photo_id']} could not be preserved: "
                f"{failed.reason}"
            ) from failed
        excluded_ids.add(str(candidate["photo_id"]))
        exclusions.append({
            "photo_id": candidate["photo_id"],
            "sequence_id": candidate["sequence_id"],
            "country": candidate["country"],
            "scene_category": candidate["scene_category"],
            "kind": failed.kind,
            "reason": failed.reason,
        })
        _write_json(paths["exclusions"], {
            "dataset_id": dataset_id,
            "exclusions": exclusions,
        })
        print(
            f"Excluding {candidate['country']} candidate {candidate['photo_id']}: "
            f"{failed.reason}",
            flush=True,
        )

    if paths["images"].exists():
        shutil.rmtree(paths["images"])
    staging.rename(paths["images"])

    category_counts = Counter(item["scene_category"] for item in selected)
    manifest = {
        "dataset_id": dataset_id,
        "dataset": "KartaView evidence-bearing four-view geolocation benchmark",
        "difficulty": "evidence_balanced",
        "experiment_name": experiment_name,
        "license": LICENSE_NAME,
        "license_url": LICENSE_URL,
        "source_url": "https://kartaview.org/",
        "sampling_metadata": {
            "probe_source": "OpenStreetMap via Overpass API",
            "probe_license": "ODbL 1.0",
            "scene_category_counts": dict(sorted(category_counts.items())),
            "maximum_target_distance_from_feature_km": (
                MAX_TARGET_DISTANCE_FROM_FEATURE_KM
            ),
            "telemetry_crop_fraction": TELEMETRY_CROP_FRACTION,
            "random_seed": random_seed,
            "selection_exclusions": exclusions,
            "manual_review_required": True,
        },
        "locations": locations,
    }
    _write_json(paths["manifest"], manifest)
    _write_json(paths["attribution"], {
        "dataset_id": dataset_id,
        "images": [
            {
                "derived_image": path,
                "source_photo_id": photo["photo_id"],
                "source_image_url": photo["source_image_url"],
                "contributor_user_id": candidate["contributor_user_id"],
                "license": LICENSE_NAME,
                "license_url": LICENSE_URL,
                "transformation": (
                    f"bottom {TELEMETRY_CROP_FRACTION:.0%} cropped, resized to fit "
                    f"within {view_size}x{view_size}, and JPEG re-encoded"
                ),
            }
            for candidate, location in zip(selected, locations)
            for photo, path in zip(candidate["photos"], location["image_paths"])
        ],
    })
    _write_review(paths, manifest, previous_review, previous_manifest)
    print(f"Prepared {len(locations)} evidence locations in {paths['images']}")
    print(f"Review page: {paths['review_html']}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discover-probes", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--refresh-review", action="store_true")
    parser.add_argument(
        "--apply-review",
        action="store_true",
        help=(
            "Preserve keep/pending locations, exclude rejected locations, and "
            "replace only their vacated scene-category slots during download."
        ),
    )
    parser.add_argument(
        "--restart-checkpoints",
        action="store_true",
        help=(
            "Discard the selected dataset ID's private probe, candidate, and "
            "exclusion checkpoints before running. Existing published images "
            "and manifests remain in place until a new download succeeds."
        ),
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--limit", type=int, default=DEFAULT_LOCATION_LIMIT)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    parser.add_argument("--feature-radius-km", type=float, default=DEFAULT_FEATURE_RADIUS_KM)
    parser.add_argument(
        "--min-feature-distance-km",
        type=float,
        default=MIN_FEATURE_DISTANCE_FROM_ANCHOR_KM,
    )
    parser.add_argument(
        "--features-per-seed-category",
        type=int,
        default=DEFAULT_FEATURES_PER_SEED_CATEGORY,
    )
    parser.add_argument(
        "--kartaview-radius",
        type=int,
        default=DEFAULT_KARTAVIEW_RADIUS_METERS,
    )
    parser.add_argument("--view-size", type=int, default=1280)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=DEFAULT_MAX_ANCHORS,
        help="Limit OSM discovery to this many seed anchors (deterministic "
             "subset). 0 or a negative value uses every seed.",
    )
    args = parser.parse_args()

    if not any((args.discover_probes, args.audit, args.download, args.refresh_review)):
        parser.error("Choose at least one preparation stage.")
    if args.apply_review and not args.download:
        parser.error("--apply-review must be used together with --download.")
    if min(
        args.limit,
        args.pool_size,
        args.features_per_seed_category,
        args.kartaview_radius,
        args.view_size,
    ) <= 0 or args.feature_radius_km <= 0 or args.min_feature_distance_km < 0:
        parser.error("All numeric preparation arguments must be positive.")

    max_anchors = args.max_anchors if args.max_anchors else None

    paths = _paths(args.dataset_id)
    if args.restart_checkpoints:
        for key in ("probes", "candidates", "exclusions"):
            paths[key].unlink(missing_ok=True)
        print(
            f"Restarted private preparation checkpoints for dataset "
            f"{args.dataset_id!r}; published outputs were preserved.",
            flush=True,
        )
    session = _preparation_session()

    probes = (
        discover_osm_evidence_probes(
            session,
            paths["probes"],
            args.feature_radius_km,
            args.features_per_seed_category,
            args.random_seed,
            args.dataset_id,
            args.min_feature_distance_km,
            max_anchors,
        )
        if args.discover_probes
        else (_read_json(paths["probes"]).get("probes", []) if args.audit else [])
    )
    candidates = (
        audit_kartaview_evidence(
            session,
            probes,
            paths["candidates"],
            args.kartaview_radius,
            args.pool_size,
            args.dataset_id,
        )
        if args.audit
        else (
            _read_json(paths["candidates"]).get("candidates", [])
            if args.download
            else []
        )
    )
    if args.download:
        prepare_evidence_dataset(
            session,
            candidates,
            paths,
            args.limit,
            args.view_size,
            args.random_seed,
            args.dataset_id,
            args.experiment_name,
            args.apply_review,
        )
    if args.refresh_review and not args.download:
        _write_review(paths, _read_json(paths["manifest"]))
        print(f"Refreshed {paths['review_html']}")


if __name__ == "__main__":
    main()
