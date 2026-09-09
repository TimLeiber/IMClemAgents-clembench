"""Audit KartaView panorama coverage and prepare four-view game instances."""

import argparse
import base64
import json
import math
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError


GAME_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = GAME_DIR.parent
RESOURCES_DIR = GAME_DIR / "resources"
SEEDS_PATH = RESOURCES_DIR / "location_seeds.json"
PHOTO_ENDPOINT = "https://api.openstreetcam.org/2.0/photo/"
LICENSE_NAME = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
DEFAULT_LOCATION_LIMIT = 30
DEFAULT_CANDIDATE_POOL_SIZE = 300
DEFAULT_DATASET_ID = "hard_v1"
DEFAULT_EXPERIMENT_NAME = "kartaview_hard_4view"
DEFAULT_PROBES_PER_SEED_PER_BAND = 4
DEFAULT_QUERY_RADIUS_METERS = 1000
DEFAULT_RANDOM_SEED = 20260901
MAX_EXPANDED_SEQUENCES = 8
MAX_TARGET_DISTANCE_FROM_PROBE_KM = 1.25
SAMPLING_BANDS = [
    {"name": "peripheral", "minimum_km": 15.0, "maximum_km": 40.0},
    {"name": "secondary_town", "minimum_km": 40.0, "maximum_km": 100.0},
    {"name": "rural_interurban", "minimum_km": 100.0, "maximum_km": 200.0},
]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _dataset_paths(dataset_id: str) -> dict[str, Path]:
    """Return isolated paths for one prepared dataset variant."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", dataset_id):
        raise ValueError(
            "dataset_id must contain only lowercase letters, digits, underscores, and hyphens"
        )

    return {
        "candidates": RESOURCES_DIR / f"kartaview_candidates_{dataset_id}.json",
        "manifest": RESOURCES_DIR / f"dataset_manifest_{dataset_id}.json",
        "attribution": RESOURCES_DIR / f"attribution_{dataset_id}.json",
        "download_exclusions": (
            RESOURCES_DIR / f"download_exclusions_{dataset_id}.json"
        ),
        "images": RESOURCES_DIR / f"images_{dataset_id}",
    }


def _destination_point(latitude: float,
                       longitude: float,
                       distance_km: float,
                       bearing_degrees: float) -> tuple[float, float]:
    """Move along a great-circle bearing by an exact distance."""
    angular_distance = distance_km / 6371.009
    latitude_radians = math.radians(latitude)
    longitude_radians = math.radians(longitude)
    bearing_radians = math.radians(bearing_degrees)

    target_latitude = math.asin(
        math.sin(latitude_radians) * math.cos(angular_distance)
        + math.cos(latitude_radians)
        * math.sin(angular_distance)
        * math.cos(bearing_radians)
    )
    target_longitude = longitude_radians + math.atan2(
        math.sin(bearing_radians)
        * math.sin(angular_distance)
        * math.cos(latitude_radians),
        math.cos(angular_distance)
        - math.sin(latitude_radians) * math.sin(target_latitude),
    )

    normalized_longitude = (
        math.degrees(target_longitude) + 540.0
    ) % 360.0 - 180.0
    return math.degrees(target_latitude), normalized_longitude


def _sample_probe(seed: dict,
                  band: dict,
                  rng: random.Random,
                  probe_index: int) -> dict:
    """Sample an area-uniform point in one annulus around a coverage anchor."""
    minimum_squared = float(band["minimum_km"]) ** 2
    maximum_squared = float(band["maximum_km"]) ** 2
    planned_distance = math.sqrt(rng.uniform(minimum_squared, maximum_squared))
    bearing = rng.uniform(0.0, 360.0)
    latitude, longitude = _destination_point(
        float(seed["latitude"]),
        float(seed["longitude"]),
        planned_distance,
        bearing,
    )
    return {
        "seed_name": seed["name"],
        "country": seed["country"],
        "country_code": seed["country_code"],
        "anchor_latitude": float(seed["latitude"]),
        "anchor_longitude": float(seed["longitude"]),
        "sampling_band": band["name"],
        "probe_index": probe_index,
        "planned_distance_from_anchor_km": planned_distance,
        "bearing_degrees": bearing,
        "latitude": latitude,
        "longitude": longitude,
    }


def _probe_plan(seeds: list[dict],
                probes_per_seed_per_band: int,
                random_seed: int) -> list[dict]:
    """Build and deterministically shuffle all geographic probe attempts."""
    rng = random.Random(random_seed)
    probes = []

    for seed in sorted(seeds, key=lambda item: (item["country_code"], item["name"])):
        for band in SAMPLING_BANDS:
            for probe_index in range(probes_per_seed_per_band):
                probes.append(_sample_probe(seed, band, rng, probe_index))

    rng.shuffle(probes)
    return probes


def _download_image(session: requests.Session, source_url: str) -> bytes:
    """Download one public KartaView image with storage/CDN fallbacks."""
    encoded_url = base64.urlsafe_b64encode(source_url.encode("utf-8")).decode("ascii").rstrip("=")
    urls = [
        source_url,
        f"https://cdn.kartaview.org/pr:sharp/{encoded_url}",
    ]

    if "/proc/" in source_url:
        large_thumbnail_url = source_url.replace("/proc/", "/lth/")
        encoded_thumbnail_url = (
            base64.urlsafe_b64encode(large_thumbnail_url.encode("utf-8"))
            .decode("ascii")
            .rstrip("=")
        )
        urls.extend([
            large_thumbnail_url,
            f"https://cdn.kartaview.org/pr:sharp/{encoded_thumbnail_url}",
        ])

    last_error = None

    for url in dict.fromkeys(urls):
        for attempt in range(2):
            try:
                response = session.get(url, timeout=(30, 180))
                response.raise_for_status()

                if not response.content:
                    raise ValueError("Image response was empty")

                return response.content
            except (requests.RequestException, ValueError) as error:
                last_error = error
                time.sleep(1 + attempt)

    raise RuntimeError(f"Could not download KartaView image {source_url}") from last_error


def _haversine_km(latitude_a: float,
                  longitude_a: float,
                  latitude_b: float,
                  longitude_b: float) -> float:
    latitude_a = math.radians(latitude_a)
    latitude_b = math.radians(latitude_b)
    latitude_delta = latitude_b - latitude_a
    longitude_delta = math.radians(longitude_b - longitude_a)
    value = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_a)
        * math.cos(latitude_b)
        * math.sin(longitude_delta / 2) ** 2
    )
    return 6371.009 * 2 * math.asin(min(1.0, math.sqrt(value)))


def _eligible_photo(photo: dict, expected_country_code: str | None) -> bool:
    sequence = photo.get("sequence") or {}

    return (
        photo.get("projection") == "PLANE"
        and photo.get("status") == "active"
        and photo.get("visibility") in {None, "public"}
        and isinstance(photo.get("fileurlProc"), str)
        and (
            expected_country_code is None
            or sequence.get("countryCode") == expected_country_code
        )
    )


def _sequence_burst(photos: list[dict], seed: dict) -> list[dict] | None:
    """Choose the compact four-frame window nearest a sampled probe."""
    by_sequence = {}

    for photo in photos:
        if not _eligible_photo(photo, seed.get("country_code")):
            continue

        sequence_id = str((photo.get("sequence") or {}).get("id", ""))

        if sequence_id:
            by_sequence.setdefault(sequence_id, []).append(photo)

    windows = []

    for sequence_photos in by_sequence.values():
        try:
            sequence_photos.sort(key=lambda photo: int(photo["sequenceIndex"]))
        except (KeyError, TypeError, ValueError):
            continue

        for start in range(len(sequence_photos) - 3):
            window = sequence_photos[start:start + 4]
            indices = [int(photo["sequenceIndex"]) for photo in window]

            # Avoid combining isolated frames merely because the nearby query
            # happened to return them from the same long sequence.
            if indices[-1] - indices[0] > 12:
                continue

            coordinates = [
                (float(photo["lat"]), float(photo["lng"]))
                for photo in window
            ]
            target_latitude, target_longitude = coordinates[1]
            span_km = max(
                _haversine_km(
                    target_latitude,
                    target_longitude,
                    latitude,
                    longitude,
                )
                for latitude, longitude in coordinates
            )

            if span_km > 0.25:
                continue

            seed_distance = _haversine_km(
                seed["latitude"],
                seed["longitude"],
                target_latitude,
                target_longitude,
            )
            windows.append((seed_distance, span_km, window))

    return min(windows, key=lambda item: (item[0], item[1]))[2] if windows else None


def _expanded_sequence_burst(session: requests.Session,
                             nearby_photos: list[dict],
                             seed: dict) -> list[dict] | None:
    """Expand nearby sequence representatives into frames around the seed."""
    burst = _sequence_burst(nearby_photos, seed)

    if burst is not None:
        return burst

    representatives = [
        photo for photo in nearby_photos
        if _eligible_photo(photo, seed.get("country_code"))
    ]
    representatives.sort(
        key=lambda photo: _haversine_km(
            seed["latitude"],
            seed["longitude"],
            float(photo["lat"]),
            float(photo["lng"]),
        )
    )
    seen_sequences = set()

    for representative in representatives:
        sequence = representative.get("sequence") or {}
        sequence_id = str(sequence.get("id", ""))

        if not sequence_id or sequence_id in seen_sequences:
            continue

        seen_sequences.add(sequence_id)

        if len(seen_sequences) > MAX_EXPANDED_SEQUENCES:
            break

        try:
            sequence_index = int(representative["sequenceIndex"])
        except (KeyError, TypeError, ValueError):
            continue

        page = sequence_index // 150 + 1
        pages = {page}
        page_offset = sequence_index % 150

        if page_offset < 3 and page > 1:
            pages.add(page - 1)

        if page_offset > 146:
            pages.add(page + 1)

        expanded = []

        for page_number in sorted(pages):
            try:
                response = session.get(
                    PHOTO_ENDPOINT,
                    params={
                        "sequenceId": sequence_id,
                        "page": page_number,
                        "itemsPerPage": 150,
                    },
                    timeout=(20, 35),
                )
                response.raise_for_status()
                sequence_photos = response.json().get("result", {}).get("data", [])
            except (requests.RequestException, ValueError):
                continue

            for photo in sequence_photos:
                photo = dict(photo)
                photo["sequence"] = sequence
                expanded.append(photo)

        burst = _sequence_burst(expanded, seed)

        if burst is not None:
            return burst

    return None


def _sampling_band(name: str) -> dict:
    for band in SAMPLING_BANDS:
        if band["name"] == name:
            return band
    raise KeyError(f"Unknown sampling band: {name}")


def _distance_is_in_band(distance_km: float, band: dict) -> bool:
    return float(band["minimum_km"]) <= distance_km <= float(band["maximum_km"])


def _final_band_quotas(limit: int) -> dict[str, int]:
    """Allocate 40% peripheral and 30% to each harder distance band."""
    weights = {
        "peripheral": 0.40,
        "secondary_town": 0.30,
        "rural_interurban": 0.30,
    }
    exact = {name: limit * weight for name, weight in weights.items()}
    quotas = {name: int(value) for name, value in exact.items()}
    remainder = limit - sum(quotas.values())
    order = sorted(
        weights,
        key=lambda name: (-(exact[name] - quotas[name]), -weights[name], name),
    )
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def audit_coverage(session: requests.Session,
                   output_path: Path,
                   radius: int = DEFAULT_QUERY_RADIUS_METERS,
                   target_count: int | None = DEFAULT_CANDIDATE_POOL_SIZE,
                   probes_per_seed_per_band: int = DEFAULT_PROBES_PER_SEED_PER_BAND,
                   random_seed: int = DEFAULT_RANDOM_SEED,
                   dataset_id: str = DEFAULT_DATASET_ID) -> list[dict]:
    """Discover a deterministic pool around random annular probes.

    Seed coordinates are coverage anchors only. They are never queried directly
    and are not candidate targets.
    """
    candidates = []
    seen_sequences = set()
    rejection_counts = Counter()
    seeds = _read_json(SEEDS_PATH)
    probes = _probe_plan(seeds, probes_per_seed_per_band, random_seed)

    for attempt_number, probe in enumerate(probes, start=1):
        params = {
            "lat": probe["latitude"],
            "lng": probe["longitude"],
            "zoomLevel": 18,
            "radius": radius,
            "join": "sequence",
            "orderBy": "id",
            "orderDirection": "desc",
        }

        try:
            response = session.get(PHOTO_ENDPOINT, params=params, timeout=(20, 35))
            response.raise_for_status()
            photos = response.json().get("result", {}).get("data", [])
        except (requests.RequestException, ValueError) as error:
            rejection_counts["api_error"] += 1
            print(
                f"probe {attempt_number}/{len(probes)} "
                f"{probe['seed_name']} {probe['sampling_band']}: API error: {error}",
                flush=True,
            )
            continue

        burst = _expanded_sequence_burst(session, photos, probe)

        if burst is None:
            rejection_counts["no_four_frame_sequence"] += 1
            continue

        photo = burst[1]
        sequence = photo.get("sequence") or {}
        sequence_id = str(sequence.get("id", ""))

        if sequence_id in seen_sequences:
            rejection_counts["duplicate_sequence"] += 1
            continue

        target_latitude = float(photo["lat"])
        target_longitude = float(photo["lng"])
        anchor_distance = _haversine_km(
            probe["anchor_latitude"],
            probe["anchor_longitude"],
            target_latitude,
            target_longitude,
        )
        band = _sampling_band(probe["sampling_band"])

        # A large KartaView query radius can move the selected sequence across
        # a band boundary. Reject it rather than mislabelling the difficulty.
        if not _distance_is_in_band(anchor_distance, band):
            rejection_counts["outside_sampling_band"] += 1
            continue

        probe_distance = _haversine_km(
            probe["latitude"],
            probe["longitude"],
            target_latitude,
            target_longitude,
        )
        if probe_distance > MAX_TARGET_DISTANCE_FROM_PROBE_KM:
            rejection_counts["too_far_from_probe"] += 1
            continue

        frame_span = max(
            _haversine_km(
                target_latitude,
                target_longitude,
                float(item["lat"]),
                float(item["lng"]),
            )
            for item in burst
        )
        candidate = {
            "seed_name": probe["seed_name"],
            "country": probe["country"],
            "country_code": probe["country_code"],
            "sampling_band": probe["sampling_band"],
            "probe_index": probe["probe_index"],
            "probe_latitude": probe["latitude"],
            "probe_longitude": probe["longitude"],
            "probe_bearing_degrees": probe["bearing_degrees"],
            "planned_distance_from_anchor_km": probe[
                "planned_distance_from_anchor_km"
            ],
            "distance_from_anchor_km": anchor_distance,
            "distance_from_probe_km": probe_distance,
            "frame_span_km": frame_span,
            "photo_id": str(photo["id"]),
            "sequence_id": sequence_id,
            "contributor_user_id": str(sequence.get("userId", "")),
            "latitude": target_latitude,
            "longitude": target_longitude,
            "heading": float(photo.get("heading") or 0),
            "shot_date": photo.get("shotDate"),
            "source_photo_url": (
                "https://kartaview.org/details/"
                f"{sequence_id}/{photo.get('sequenceIndex')}/track-info"
            ),
            "photos": [
                {
                    "photo_id": str(item["id"]),
                    "sequence_index": int(item["sequenceIndex"]),
                    "latitude": float(item["lat"]),
                    "longitude": float(item["lng"]),
                    "source_image_url": item["fileurlProc"],
                }
                for item in burst
            ],
        }
        candidates.append(candidate)
        seen_sequences.add(sequence_id)
        print(
            f"candidate {len(candidates)}: {probe['country']} "
            f"{probe['sampling_band']} at {anchor_distance:.1f} km from "
            f"{probe['seed_name']} ({probe_distance:.2f} km from probe)",
            flush=True,
        )
        time.sleep(0.10)

        if target_count is not None and len(candidates) >= target_count:
            break

    document = {
        "dataset_id": dataset_id,
        "sampling": {
            "strategy": "deterministic_area_uniform_annular_probes",
            "random_seed": random_seed,
            "query_radius_meters": radius,
            "maximum_target_distance_from_probe_km": (
                MAX_TARGET_DISTANCE_FROM_PROBE_KM
            ),
            "probes_per_seed_per_band": probes_per_seed_per_band,
            "bands": SAMPLING_BANDS,
            "probe_attempts": min(len(probes), attempt_number if probes else 0),
            "rejections": dict(sorted(rejection_counts.items())),
        },
        "candidates": candidates,
    }
    _write_json(output_path, document)
    print(
        f"Found {len(candidates)} eligible locations; wrote {output_path}",
        flush=True,
    )
    return candidates


def select_balanced_candidates(candidates: list[dict],
                               limit: int,
                               random_seed: int) -> list[dict]:
    """Select distinct locations using the documented per-band allocation."""
    eligible_candidates = [
        candidate for candidate in candidates
        if float(candidate.get("distance_from_probe_km", 0.0))
        <= MAX_TARGET_DISTANCE_FROM_PROBE_KM
    ]
    if len(eligible_candidates) < limit:
        raise RuntimeError(
            f"Only {len(eligible_candidates)} eligible locations are available after "
            f"enforcing the {MAX_TARGET_DISTANCE_FROM_PROBE_KM:.2f} km probe-distance "
            f"limit; requested {limit}."
        )

    band_names = [band["name"] for band in SAMPLING_BANDS]
    quotas = _final_band_quotas(limit)
    candidates_by_band = defaultdict(list)
    for candidate in eligible_candidates:
        band_name = candidate.get("sampling_band")
        if band_name in quotas:
            candidates_by_band[band_name].append(candidate)

    candidate_counts = {
        band_name: len(candidates_by_band[band_name])
        for band_name in band_names
    }
    insufficient = {
        band_name: (candidate_counts[band_name], quotas[band_name])
        for band_name in band_names
        if candidate_counts[band_name] < quotas[band_name]
    }
    if insufficient:
        raise RuntimeError(
            "Candidate pool cannot satisfy the per-band location quotas: "
            f"{insufficient}. Run the audit with a larger pool before downloading."
        )

    rng = random.Random(random_seed)
    selected = []
    for band_name in band_names:
        choices = sorted(
            candidates_by_band[band_name],
            key=lambda candidate: (
                candidate["sequence_id"],
                candidate["photo_id"],
            ),
        )
        selected.extend(rng.sample(choices, quotas[band_name]))

    rng.shuffle(selected)
    return selected


class CandidateImageDownloadError(RuntimeError):
    """Report one unusable KartaView candidate without losing the whole run."""

    def __init__(self, candidate: dict, view_index: int, error: Exception):
        self.candidate = candidate
        self.view_index = view_index
        self.original_error = error
        super().__init__(
            f"Could not materialize {candidate['country']} candidate "
            f"{candidate['photo_id']} view {view_index}: {error}"
        )


def _materialize_selected_locations(session: requests.Session,
                                    selected: list[dict],
                                    staging_dir: Path,
                                    images_directory_name: str,
                                    view_size: int,
                                    image_cache: dict[str, bytes]) -> list[dict]:
    """Download and render one complete selected set into a staging tree."""
    locations = []

    for location_index, candidate in enumerate(selected, start=1):
        location_id = f"location_{location_index:03d}"
        location_dir = staging_dir / location_id
        location_dir.mkdir()
        print(f"Preparing {location_id} ({candidate['country']})", flush=True)
        image_paths = []

        for view_index, photo in enumerate(candidate["photos"]):
            source_url = photo["source_image_url"]
            try:
                image_bytes = image_cache.get(source_url)
                if image_bytes is None:
                    image_bytes = _download_image(session, source_url)
                    image_cache[source_url] = image_bytes
                with Image.open(BytesIO(image_bytes)) as source_image:
                    image = source_image.convert("RGB")
                    image.thumbnail((view_size, view_size), Image.Resampling.LANCZOS)
            except (RuntimeError, UnidentifiedImageError, OSError) as error:
                raise CandidateImageDownloadError(
                    candidate,
                    view_index,
                    error,
                ) from error

            filename = f"view_{view_index:02d}.jpg"
            output_path = location_dir / filename
            image.save(
                output_path,
                format="JPEG",
                quality=92,
                optimize=True,
            )
            relative_path = output_path.relative_to(staging_dir)
            game_path = (
                Path("geolocate/resources")
                / images_directory_name
                / relative_path
            )
            image_paths.append(game_path.as_posix())

        locations.append({
            "location_id": location_id,
            "image_paths": image_paths,
            "latitude": candidate["latitude"],
            "longitude": candidate["longitude"],
            "country": candidate["country"],
            "country_code": candidate["country_code"],
            "source_photo_id": candidate["photo_id"],
            "source_sequence_id": candidate["sequence_id"],
            "shot_date": candidate["shot_date"],
            "sampling_band": candidate["sampling_band"],
            "distance_from_anchor_km": candidate["distance_from_anchor_km"],
            "frame_span_km": candidate["frame_span_km"],
        })

    return locations


def prepare_dataset(session: requests.Session,
                    candidates: list[dict],
                    limit: int,
                    view_size: int,
                    paths: dict[str, Path],
                    dataset_id: str,
                    experiment_name: str,
                    random_seed: int) -> list[dict]:
    """Download one balanced hard set into isolated, versioned paths."""
    images_dir = paths["images"]
    staging_dir = images_dir.parent / f"{images_dir.name}.staging"
    exclusions_path = paths.get(
        "download_exclusions",
        paths["manifest"].with_name(f"download_exclusions_{dataset_id}.json"),
    )
    exclusions_document = (
        _read_json(exclusions_path)
        if exclusions_path.exists()
        else {"dataset_id": dataset_id, "download_exclusions": []}
    )
    download_exclusions = exclusions_document.get("download_exclusions", [])
    excluded_photo_ids = {
        str(exclusion["photo_id"])
        for exclusion in download_exclusions
    }
    image_cache = {}

    while True:
        shutil.rmtree(staging_dir, ignore_errors=True)
        selectable_candidates = [
            candidate for candidate in candidates
            if str(candidate["photo_id"]) not in excluded_photo_ids
        ]
        selected = select_balanced_candidates(
            selectable_candidates,
            limit,
            random_seed,
        )
        staging_dir.mkdir(parents=True)

        try:
            locations = _materialize_selected_locations(
                session,
                selected,
                staging_dir,
                images_dir.name,
                view_size,
                image_cache,
            )
        except CandidateImageDownloadError as error:
            candidate = error.candidate
            excluded_photo_ids.add(str(candidate["photo_id"]))
            exclusion = {
                "photo_id": candidate["photo_id"],
                "sequence_id": candidate["sequence_id"],
                "country": candidate["country"],
                "sampling_band": candidate["sampling_band"],
                "failed_view_index": error.view_index,
                "reason": str(error.original_error),
            }
            download_exclusions.append(exclusion)
            _write_json(exclusions_path, {
                "dataset_id": dataset_id,
                "download_exclusions": download_exclusions,
            })
            print(
                f"Excluding unavailable {candidate['country']} candidate "
                f"{candidate['photo_id']} and reselecting the complete set.",
                flush=True,
            )
            continue
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        break

    if images_dir.exists():
        shutil.rmtree(images_dir)
    staging_dir.rename(images_dir)

    band_counts = Counter(candidate["sampling_band"] for candidate in selected)
    country_count = len({candidate["country_code"] for candidate in selected})
    _write_json(paths["manifest"], {
        "dataset_id": dataset_id,
        "dataset": "KartaView four-view geolocation benchmark",
        "difficulty": "hard",
        "experiment_name": experiment_name,
        "license": LICENSE_NAME,
        "license_url": LICENSE_URL,
        "source_url": "https://kartaview.org/",
        "selection": {
            "random_seed": random_seed,
            "country_count": country_count,
            "maximum_locations_per_country": max(
                Counter(candidate["country_code"] for candidate in selected).values()
            ),
            "maximum_target_distance_from_probe_km": (
                MAX_TARGET_DISTANCE_FROM_PROBE_KM
            ),
            "band_allocation": "40_percent_peripheral_30_percent_each_harder_band",
            "band_counts": dict(sorted(band_counts.items())),
            "download_exclusions": download_exclusions,
        },
        "locations": locations,
    })
    _write_json(
        paths["attribution"],
        {
            "dataset_id": dataset_id,
            "images": attribution_records(selected, view_size, images_dir.name),
        },
    )
    print(f"Prepared {len(locations)} locations in {images_dir}")
    print(f"Band counts: {dict(sorted(band_counts.items()))}")
    return selected


def attribution_records(candidates: list[dict],
                        view_size: int,
                        images_directory_name: str) -> list[dict]:
    """Build exact per-frame attribution for already selected candidates."""
    records = []

    for location_index, candidate in enumerate(candidates, start=1):
        location_id = f"location_{location_index:03d}"

        for view_index, photo in enumerate(candidate["photos"]):
            derived_image = (
                Path("geolocate/resources")
                / images_directory_name
                / location_id
                / f"view_{view_index:02d}.jpg"
            )
            records.append({
                "derived_image": derived_image.as_posix(),
                "source_photo_id": photo["photo_id"],
                "source_photo_url": (
                    "https://kartaview.org/details/"
                    f"{candidate['sequence_id']}/{photo['sequence_index']}/track-info"
                ),
                "source_image_url": photo["source_image_url"],
                "contributor_user_id": candidate["contributor_user_id"],
                "license": LICENSE_NAME,
                "license_url": LICENSE_URL,
                "transformation": (
                    f"resized to fit within {view_size}x{view_size} and JPEG re-encoded"
                ),
            })

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--refresh-attribution", action="store_true")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--limit", type=int, default=DEFAULT_LOCATION_LIMIT)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    parser.add_argument("--radius", type=int, default=DEFAULT_QUERY_RADIUS_METERS)
    parser.add_argument(
        "--probes-per-seed-per-band",
        type=int,
        default=DEFAULT_PROBES_PER_SEED_PER_BAND,
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--view-size", type=int, default=1280)
    args = parser.parse_args()

    if not args.audit and not args.download and not args.refresh_attribution:
        parser.error("Select --audit, --download, --refresh-attribution, or a combination.")

    if args.limit <= 0 or args.pool_size <= 0:
        parser.error("--limit and --pool-size must be positive")
    if args.pool_size < args.limit:
        parser.error("--pool-size must be at least --limit")
    if args.radius <= 0 or args.probes_per_seed_per_band <= 0:
        parser.error("--radius and --probes-per-seed-per-band must be positive")

    paths = _dataset_paths(args.dataset_id)
    session = requests.Session()
    session.headers["User-Agent"] = "clembench-geolocate-dataset-preparation/1.0"
    candidates = audit_coverage(
        session,
        output_path=paths["candidates"],
        radius=args.radius,
        target_count=args.pool_size,
        probes_per_seed_per_band=args.probes_per_seed_per_band,
        random_seed=args.random_seed,
        dataset_id=args.dataset_id,
    ) if args.audit else (
        _read_json(paths["candidates"]).get("candidates", [])
    )

    if args.download:
        prepare_dataset(
            session,
            candidates,
            args.limit,
            args.view_size,
            paths,
            args.dataset_id,
            args.experiment_name,
            args.random_seed,
        )

    if args.refresh_attribution and not args.download:
        selected = select_balanced_candidates(
            candidates,
            args.limit,
            args.random_seed,
        )
        _write_json(
            paths["attribution"],
            {
                "dataset_id": args.dataset_id,
                "images": attribution_records(
                    selected,
                    args.view_size,
                    paths["images"].name,
                ),
            },
        )
        print(f"Refreshed {paths['attribution']}")


if __name__ == "__main__":
    main()
