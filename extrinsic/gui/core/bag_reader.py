"""Reading ROS bags without ROS: the LiDAR side (sweeps around a moment).

Opening a bag, listing its topics and seeking (`BagSource`) are shared with the
Camera Intrinsic tool and live in shared.ros.bag; names are re-exported here.
"""

from __future__ import annotations

from shared.ros.bag import (  # noqa: F401
    IMAGE_TYPES,
    LIDAR_TYPES,
    BagInfo,
    BagSource,
    TopicInfo,
    inspect,
    open_reader,
)


def sweep_window(stamps: list[int], t_ns: int, frames: int) -> tuple[int, int]:
    """[lo, hi) indices of the `frames` sorted stamps centred on t_ns.

    Shifted rather than truncated at either end of the recording, so the count
    asked for is the count returned whenever the bag holds that many.
    """
    want = max(int(frames), 1)
    centre = min(range(len(stamps)), key=lambda i: abs(stamps[i] - t_ns))
    lo = max(0, centre - want // 2)
    hi = min(len(stamps), lo + want)
    lo = max(0, hi - want)
    return lo, hi


def sweeps_around(src: "BagSource", topic: str, t_ns: int, frames: int = 5, progress=None):
    """The individual sweeps nearest t_ns, newest-in-the-middle order kept.

    Split out from `accumulate_cloud` so a caller can try one sweep, then three,
    then five without reading the bag again. The window grows symmetrically about
    the centre sweep, so the smaller counts are exactly the middle of the larger
    ones -- trying several depths costs one read, not one read per depth.

    Returns (chunks, intensities, rings); the lists line up and any of the last
    two may hold None where the topic does not carry that field.
    """
    import numpy as np

    from gui.core.decode import cloud_extra, cloud_to_struct

    stamps = sorted(src.timestamps(topic, progress=progress))
    if not stamps:
        return [], [], []

    lo, hi = sweep_window(stamps, t_ns, frames)

    chunks, intensities, rings = [], [], []
    for _, msg in src.range(topic, stamps[lo], stamps[hi - 1] + 1):
        rec = cloud_to_struct(msg)
        xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float32)
        finite = np.isfinite(xyz).all(axis=1)
        chunks.append(xyz[finite])
        inten = cloud_extra(msg, "intensity")
        intensities.append(None if inten is None else np.asarray(inten)[finite].astype(np.float32))
        rg = cloud_extra(msg, "ring")
        rings.append(None if rg is None else np.asarray(rg)[finite].astype(np.int32))
    return chunks, intensities, rings


def stack_sweeps(chunks, intensities, rings):
    """Glue sweeps into one cloud, as `accumulate_cloud` returns them."""
    import numpy as np

    if not chunks:
        empty = np.empty((0, 3), np.float32)
        return empty, 0, empty, None, None
    mid = len(chunks) // 2
    intensity = np.concatenate(intensities) if all(i is not None for i in intensities) else None
    # Sweeps are stacked, so ring numbers repeat. Offset each sweep so a walk
    # along "one ring" does not jump between sweeps.
    if all(r is not None for r in rings):
        span = max(int(r.max()) + 1 for r in rings)
        ring = np.concatenate([r + i * span for i, r in enumerate(rings)])
    else:
        ring = None
    return np.vstack(chunks), len(chunks), chunks[mid], intensity, ring


def middle_slice(items, want: int):
    """The middle `want` entries -- how a smaller sweep count sits inside a larger."""
    if want >= len(items):
        return list(items)
    lo = (len(items) - want) // 2
    return list(items[lo:lo + want])


def accumulate_cloud(src: "BagSource", topic: str, t_ns: int, frames: int = 5, progress=None):
    """Stack `frames` sweeps centred on t_ns.

    Counted, not timed. Sensors differ too much for a fixed window to mean the
    same thing: a merged 19 Hz cloud carries 63k points per message where a raw
    128-channel unit at 10 Hz carries 920k, so half a second is 1M points on one
    and 9M on the other.

    Returns (stacked, count, middle sweep, intensity, ring). `ring` is None when
    the topic does not carry one -- a merged cloud never does, since ring numbers
    from different sensors cannot be combined.

    The middle sweep comes back separately because point spacing must be judged
    on a single sweep; the stack's nearest-neighbour distance reports the voxel
    size instead.
    """
    return stack_sweeps(*sweeps_around(src, topic, t_ns, frames, progress))


