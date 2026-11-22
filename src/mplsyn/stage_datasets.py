"""Generate every dataset used by the modeling notebooks into a staging folder.

Run it with ``pixi run -e dev stage-datasets`` (or ``python -m
mplsyn.stage_datasets [DIRECTORY]``). The ``.npz`` files written to the staging
folder are exactly what the ``load_*`` functions look for, so the folder can be
uploaded as-is as the assets of a GitHub release, and tried locally with::

    MPLSYN_DATA_HOME=staging jupyter lab notebooks

Datasets already present in the folder are kept, so reruns are cheap; delete the
folder (or bump ``DATASET_VERSION``) to regenerate everything.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time

from mplsyn import datasets

#: Default staging folder, relative to the current directory (git-ignored).
DEFAULT_STAGING_DIR = "staging"

#: ``(loader, arguments)`` for every dataset requested by the notebooks. Keep the
#: arguments in sync with the notebooks: a mismatch only means that the notebook
#: generates (and caches) that dataset itself instead of using the staged file.
NOTEBOOK_DATASETS = [
    # 05_train_scene_classification
    (datasets.load_classification_data, {"n_per_class": 72, "size": 64}),
    # 06_train_semantic_segmentation
    (datasets.load_segmentation_data, {"n_samples": 360, "size": 64}),
    # 07_train_cloud_occlusion
    (datasets.load_cloud_data, {"n_samples": 360, "size": 64}),
    # 08_train_object_detection: full run, FAST run, 64 px resolution ablation
    (datasets.load_detection_data, {"n_samples": 480, "size": 96}),
    (datasets.load_detection_data, {"n_samples": 180, "size": 96}),
    (datasets.load_detection_data, {"n_samples": 200, "size": 64, "seed": 7}),
    # 08_train_object_detection: tree instance segmentation
    (datasets.load_forest_data, {"n_samples": 400, "size": 64}),
]


def _describe(loader, kwargs):
    kind = loader.__name__.removeprefix("load_").removesuffix("_data")
    return f"{kind}({', '.join(f'{k}={v}' for k, v in kwargs.items())})"


def stage_datasets(directory=DEFAULT_STAGING_DIR):
    """Generate all notebook datasets into ``directory``; return the new files."""
    directory = pathlib.Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # ``load_*`` read their cache location from the environment.
    os.environ[datasets.DATA_HOME_ENV] = str(directory)
    # No download: staging must be generated from scratch, never fetched.
    os.environ.pop(datasets.DATA_URL_ENV, None)
    datasets.DEFAULT_DATA_URL = None

    created = []
    for loader, kwargs in NOTEBOOK_DATASETS:
        before = set(directory.glob("*.npz"))
        started = time.time()
        # uint8 images: skip the float32 conversion, we only need the file.
        extra = {} if loader is datasets.load_forest_data else {"as_float32": False}
        loader(**kwargs, **extra)
        new_files = sorted(set(directory.glob("*.npz")) - before)
        created.extend(new_files)
        status = (
            f"generated in {time.time() - started:.1f} s" if new_files else "already staged")
        print(f"{_describe(loader, kwargs)}: {status}", flush=True)
    return created


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "directory", nargs="?", default=DEFAULT_STAGING_DIR,
        help=f"staging folder (default: ./{DEFAULT_STAGING_DIR})")
    args = parser.parse_args(argv)
    stage_datasets(args.directory)
    directory = pathlib.Path(args.directory)
    files = sorted(directory.glob("*.npz"))
    total = sum(path.stat().st_size for path in files)
    print(f"\n{len(files)} files, {total / 1e6:.1f} MB in {directory.resolve()}")
    for path in files:
        print(f"  {path.stat().st_size / 1e6:7.2f} MB  {path.name}")


if __name__ == "__main__":
    main()
