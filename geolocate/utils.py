"""Response parsing and geographic scoring helpers for Geolocate."""

import math
import re
from typing import Dict, List


EARTH_RADIUS_KM = 6371.009
DEFAULT_HALF_SCORE_DISTANCE_KM = 100.0
DEFAULT_CLOSE_RANGE_HALF_SCORE_KM = 5.0
COUNTRY_ALIASES_BY_CODE = {
    "US": {"usa", "unitedstatesofamerica", "america"},
    "GB": {"uk", "greatbritain", "britain"},
    "CZ": {"czechrepublic"},
    "BA": {"bosnia", "bosniaherzegovina"},
    "MK": {"macedonia"},
    "NL": {"holland"},
    "GR": {"hellenicrepublic"},
    "SK": {"slovakrepublic"},
}


def parse_geolocation_response(response: str,
                               labels: Dict[str, str]) -> dict:
    """Parse one strict four-line coordinate prediction."""
    expected = [
        ("explanation", labels["explanation_label"]),
        ("latitude", labels["latitude_label"]),
        ("longitude", labels["longitude_label"]),
        ("country", labels["country_label"]),
    ]
    lines = response.strip().splitlines()

    if len(lines) != len(expected):
        raise ValueError("Response must contain exactly four non-wrapped lines.")

    values = {}

    for line, (key, label) in zip(lines, expected):
        match = re.fullmatch(rf"{re.escape(label)}:\s*(.+?)\s*", line)

        if match is None:
            raise ValueError(f"Expected line in the form '{label}: <value>'.")

        values[key] = match.group(1)

    try:
        latitude = float(values["latitude"])
        longitude = float(values["longitude"])
    except ValueError as error:
        raise ValueError("Latitude and longitude must be decimal numbers.") from error

    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("Latitude must be a finite number from -90 to 90.")

    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("Longitude must be a finite number from -180 to 180.")

    return {
        "latitude": latitude,
        "longitude": longitude,
        "country": values["country"],
        "explanation": values["explanation"],
    }


def distance_feedback_band(distance_km: float,
                           bands: List[Dict]) -> tuple[int, str]:
    """Return the zero-based feedback band and its public label."""
    if distance_km < 0:
        raise ValueError("Distance cannot be negative.")

    if not bands:
        raise ValueError("At least one distance feedback band is required.")

    for index, band in enumerate(bands):
        upper_bound = band.get("upper_bound_km")
        label = band.get("label")

        if not isinstance(label, str) or not label:
            raise ValueError("Each distance feedback band requires a label.")

        if upper_bound is None or distance_km < float(upper_bound):
            return index, label

    raise ValueError("Distance feedback bands must end with an open upper bound.")


def haversine_distance_km(latitude_a: float,
                          longitude_a: float,
                          latitude_b: float,
                          longitude_b: float,
                          earth_radius_km: float = EARTH_RADIUS_KM) -> float:
    """Return great-circle distance between two WGS84 coordinate pairs."""
    latitude_a_rad = math.radians(latitude_a)
    latitude_b_rad = math.radians(latitude_b)
    latitude_delta = math.radians(latitude_b - latitude_a)
    longitude_delta = math.radians(longitude_b - longitude_a)
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_a_rad)
        * math.cos(latitude_b_rad)
        * math.sin(longitude_delta / 2) ** 2
    )
    central_angle = 2 * math.asin(min(1.0, math.sqrt(haversine)))
    return earth_radius_km * central_angle


def inverse_distance_quality(distance_km: float,
                             half_score_distance_km: float = DEFAULT_HALF_SCORE_DISTANCE_KM) -> float:
    """Return the primary inverse-distance quality score on a 0--100 scale."""
    if distance_km < 0:
        raise ValueError("Distance cannot be negative.")

    if half_score_distance_km <= 0:
        raise ValueError("Half-score distance must be positive.")

    return 100 * half_score_distance_km / (half_score_distance_km + distance_km)


def exponential_distance_quality(distance_km: float,
                                 half_score_distance_km: float = DEFAULT_HALF_SCORE_DISTANCE_KM) -> float:
    """Return the diagnostic exponential-decay score on a 0--100 scale."""
    if distance_km < 0:
        raise ValueError("Distance cannot be negative.")

    if half_score_distance_km <= 0:
        raise ValueError("Half-score distance must be positive.")

    return 100 * math.exp(-math.log(2) * distance_km / half_score_distance_km)


def close_range_quality(distance_km: float,
                        half_score_distance_km: float = DEFAULT_CLOSE_RANGE_HALF_SCORE_KM) -> float:
    """Return the diagnostic close-range score on a 0--100 scale.

    Exponential decay with a short half-score distance so that the metric
    separates predictions inside the 0--15 km ring where both the inverse and
    the wide exponential metric already saturate near 100. Distances beyond a
    few half-score distances contribute essentially nothing.
    """
    if distance_km < 0:
        raise ValueError("Distance cannot be negative.")

    if half_score_distance_km <= 0:
        raise ValueError("Half-score distance must be positive.")

    return 100 * math.exp(-math.log(2) * distance_km / half_score_distance_km)


def normalize_country(value: str) -> str:
    """Normalize a country name for a conservative string comparison."""
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def country_is_correct(prediction: str,
                       target_country: str,
                       target_country_code: str | None = None) -> bool:
    """Accept the target country name or its supplied ISO alpha-2 code."""
    normalized_prediction = normalize_country(prediction)
    accepted = {normalize_country(target_country)}

    if target_country_code:
        normalized_code = normalize_country(target_country_code)
        accepted.add(normalized_code)
        accepted.update(COUNTRY_ALIASES_BY_CODE.get(target_country_code.upper(), set()))

    return normalized_prediction in accepted
