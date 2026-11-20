"""Map official expert annotations to frozen spatial keyframe targets."""
import argparse
import io
from itertools import groupby
import json
from pathlib import Path
import re
from zipfile import ZipFile

import h5py
import numpy as np
import torch

from data import EpisodeDataset, _read_expert, preprocess_rgb
from goal_targets import GoalTargets, events_from_durations
from spatial_goal_targets import R3MSpatial, SHAPE, write_json


def phases(task, rows):
    if any(type(duration) is not int or duration <= 0 for _, duration in rows):
        raise ValueError(f"Invalid expert durations for {task}: {rows}")

    if task == "observe_and_pickup":
        # The expert adds the wall between its wait and pickup calls. The
        # first RGB recorded by pickup already shows the completed wait.
        (wait_text, wait_frames), (pickup_text, first_pickup_frames), *rest = rows
        if (wait_text != "Wait for wall appear." or pickup_text != "Pick up the target object."
                or first_pickup_frames < 2 or any(text != pickup_text for text, _ in rest)):
            raise ValueError(f"Unexpected wait/pickup annotations: {rows}")
        yield wait_text, wait_frames + 1, None
        yield pickup_text, first_pickup_frames - 1 + sum(duration for _, duration in rest), None
        return

    if task == "swap_blocks":
        # Each placement comprises grasp, lift, place, and return. The
        # dynamically described source position can change after placement.
        move = re.compile(r"Use (left|right) arm to pick up the (?:left|middle|right) block "
                          r"and move it to the (left|middle|right) tray\.")
        if len(rows) != 14 or any(text != "Press the button." for text, _ in rows[12:]):
            raise ValueError(f"Unexpected swap expert sequence: {rows}")
        for start in (0, 4, 8):
            placement = rows[start:start + 4]
            matches = [move.fullmatch(text) for text, _ in placement]
            if any(match is None for match in matches) or len({m.groups() for m in matches}) != 1:
                raise ValueError(f"Inconsistent arm or destination in a placement: {placement}")
            yield placement[0][0], sum(duration for _, duration in placement), None
        yield rows[12][0], sum(duration for _, duration in rows[12:]), None
        return

    for text, group in groupby(rows, key=lambda row: row[0]):
        durations = [row[1] for row in group]
        if (task in ("put_back_block", "rearrange_blocks", "blocks_ranking_try")
                and text in ("Press button.", "Press the button.")):
            # Each expert tests button contact at the end of its second move.
            # Later moves release the button and can return the arm to origin.
            expected = 3 if task == "blocks_ranking_try" else 4
            if len(durations) != expected:
                raise ValueError(f"Unexpected press motions for {task}: {durations}")
            yield text, sum(durations), sum(durations[:2]) - 1
        elif task == "press_button" and text.endswith("button one time."):
            if len(durations) % 3:
                raise ValueError(f"Expert approach/press/retract annotations are incomplete: {durations}")
            for start in range(0, len(durations), 3):
                approach, down, up = durations[start:start + 3]
                yield text, approach + down + up, approach + down - 1
        else:
            yield text, sum(durations), None


@torch.inference_mode()
def build(archive, prepared, weights, output, rmbench):
    import sys
    sys.path.insert(0, str(rmbench))
    from XPolicyLab.utils.process_data import decode_image_bit

    torch.set_num_threads(4)
    dataset = EpisodeDataset(prepared)
    task = dataset.manifest["task"]
    teacher = R3MSpatial(weights)
    output.mkdir(parents=True, exist_ok=True)
    events, features = [], []
    with ZipFile(archive) as packed:
        member, = (name for name in packed.namelist() if name.endswith("/language_annotation.json"))
        annotation = json.loads(packed.read(member))
        if set(annotation) != {f"episode_{item['id']}" for item in dataset.episodes}:
            raise ValueError(f"Official and prepared episode IDs differ for {task}")
        for item in dataset.episodes:
            ident = item["id"]
            labels = list(phases(task, annotation[f"episode_{ident}"]))
            current = events_from_durations(ident, [duration for _, duration, _ in labels],
                                            item["steps"], member)
            rgb = dataset._load(ident, "rgb")
            with h5py.File(io.BytesIO(packed.read(item["archive_member"])), "r") as h5:
                images, state, action, _ = _read_expert(h5)
                if (not np.array_equal(state, dataset._load(ident, "state"))
                        or not np.array_equal(action, dataset._load(ident, "action"))):
                    raise ValueError(f"State-action alignment changed for {task} episode {ident}")
                keyframes = []
                for event, (text, _, contact_offset) in zip(current, labels):
                    start = event["start_frame"]
                    if contact_offset is not None:
                        event["goal_frame_index"] = start + contact_offset
                        mask = event["valid_mask"]
                        mask["end_frame"] = min(mask["end_frame"], event["goal_frame_index"])
                    frame = event["goal_frame_index"]
                    image = decode_image_bit(images[frame])
                    if not np.array_equal(preprocess_rgb(image), rgb[frame]):
                        raise ValueError(f"Keyframe RGB mismatch at {task} episode {ident}, frame {frame}")
                    event.update(description=text, goal_feature=len(events))
                    events.append(event)
                    keyframes.append(image)
            for start in range(0, len(keyframes), 32):
                tensor, _ = teacher(keyframes[start:start + 32])
                features.extend(tensor.numpy())
    np.save(output / "goal_features.npy", np.stack(features), allow_pickle=False)
    write_json(output / "events.json", {"events": events, "target_shape": SHAPE,
                                       "teacher": "r3m_resnet18_layer4", "task": task})
    targets = GoalTargets(output, dataset.episodes, SHAPE)
    valid = sum(int((v >= 0).sum()) for v in targets.anchor_index.values())
    if not valid:
        raise ValueError(f"No valid keyframe labels for {task}")
    return {"task": task, "events": len(events), "valid_anchors": valid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "data", "weights", "output", "rmbench"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    info = build(args.archive, args.data, args.weights, args.output, args.rmbench)
    write_json(args.output / "complete.json", info)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
