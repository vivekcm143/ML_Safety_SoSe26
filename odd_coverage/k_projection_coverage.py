"""k-projection coverage of the CARLA test suite over the declared ODD.

Exercise 4.5. The ODD is discretised into the dimensions listed in DIMENSIONS.
k-projection coverage is the fraction of all k-way value combinations (taken over
every choice of k dimensions) that occurs in at least one test frame.

The four provided test splits fix the (weather, illumination, scene_layout) triple:

    test           -> (clear, daylight, training_town)
    test-fog       -> (fog,   daylight, training_town)
    test-night     -> (clear, night,    training_town)
    test-town-01   -> (clear, daylight, unseen_town)

The object-presence and ego-speed dimensions vary freely inside every split, so
their full cross product is realised in each of the four environments. Pass
--labels-root to verify the object-presence part against the actual labels.csv
files instead of assuming it.

Usage:
    python k_projection_coverage.py
    python k_projection_coverage.py --labels-root /path/to/data
"""

import argparse
import os
from itertools import combinations, product

DIMENSIONS = {
    "weather": ["clear", "fog", "rain"],
    "illumination": ["daylight", "low_sun_dusk", "night"],
    "scene_layout": ["training_town", "unseen_town"],
    "pedestrian_present": ["yes", "no"],
    "vehicle_present": ["yes", "no"],
    "traffic_light_present": ["yes", "no"],
    "ego_speed_kmh": ["0-20", "20-35", "35-50"],
}

SPLITS = {
    "test": ("clear", "daylight", "training_town"),
    "test-fog": ("fog", "daylight", "training_town"),
    "test-night": ("clear", "night", "training_town"),
    "test-town-01": ("clear", "daylight", "unseen_town"),
}

FREE_DIMS = ["pedestrian_present", "vehicle_present", "traffic_light_present", "ego_speed_kmh"]

LABEL_COLUMNS = {
    "pedestrian_present": "has_pedestrian",
    "vehicle_present": "has_vehicle",
    "traffic_light_present": "has_traffic_light",
}


def observed_object_combinations(labels_root, split):
    """Read labels.csv and return the presence triples actually observed."""
    import pandas as pd

    df = pd.read_csv(os.path.join(labels_root, split, "labels.csv"))
    df.columns = df.columns.str.strip()

    triples = set()
    for _, row in df.iterrows():
        triples.add(tuple(
            "yes" if bool(row[LABEL_COLUMNS[d]]) else "no"
            for d in ("pedestrian_present", "vehicle_present", "traffic_light_present")
        ))
    return triples


def build_samples(labels_root=None):
    samples = []
    for split, (weather, illumination, scene) in SPLITS.items():
        if labels_root is not None:
            triples = observed_object_combinations(labels_root, split)
        else:
            triples = set(product(*[DIMENSIONS[d] for d in FREE_DIMS[:3]]))

        for ped, veh, tl in sorted(triples):
            for speed in DIMENSIONS["ego_speed_kmh"]:
                samples.append({
                    "weather": weather,
                    "illumination": illumination,
                    "scene_layout": scene,
                    "pedestrian_present": ped,
                    "vehicle_present": veh,
                    "traffic_light_present": tl,
                    "ego_speed_kmh": speed,
                })
    return samples


def k_projection_coverage(samples, k):
    names = list(DIMENSIONS)
    total = 0
    covered = 0
    missing = []

    for subset in combinations(names, k):
        required = set(product(*[DIMENSIONS[d] for d in subset]))
        seen = {tuple(s[d] for d in subset) for s in samples}
        total += len(required)
        covered += len(required & seen)
        missing.extend((subset, combo) for combo in sorted(required - seen))

    return covered, total, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-root", default=None,
                        help="directory containing the test/ test-fog/ ... splits")
    parser.add_argument("--show-missing", action="store_true")
    args = parser.parse_args()

    samples = build_samples(args.labels_root)

    grid = 1
    for values in DIMENSIONS.values():
        grid *= len(values)

    print(f"ODD dimensions          : {len(DIMENSIONS)}")
    print(f"Full ODD grid            : {grid} cells")
    print(f"Realised configurations  : {len(samples)}")
    print()

    for k in (1, 2, 3):
        covered, total, missing = k_projection_coverage(samples, k)
        print(f"k={k}: {covered}/{total} = {covered / total:.4f}")
        if args.show_missing:
            for subset, combo in missing:
                print(f"      uncovered {dict(zip(subset, combo))}")


if __name__ == "__main__":
    main()
