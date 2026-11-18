"""Build offline spatial goal targets and audit the recorded event completions."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import h5py
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models import resnet18

from data import DEMO_COUNT, EpisodeDataset, _read_expert, preprocess_rgb
from goal_targets import GoalTargets, events_from_durations

SHAPE = (9, 512)
R3M_REVISION = "1a4f077fe01db52c8f5d9f8d6641b6e03f688420"
R3M_REPOSITORY = "https://huggingface.co/surajnair/r3m-18"
MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


class R3MSpatial(nn.Module):
    def __init__(self, filename):
        super().__init__()
        payload = torch.load(filename, map_location="cpu", weights_only=True)
        if "r3m" in payload:
            payload = payload["r3m"]
        state = {key.split("convnet.", 1)[1]: value for key, value in payload.items()
                 if key.startswith(("convnet.", "module.convnet."))}
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.backbone.load_state_dict(state, strict=True)
        self.requires_grad_(False).eval()
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def forward(self, images):
        x = self.backbone
        device = x.conv1.weight.device
        image_batch = torch.stack([self.transform(Image.fromarray(image)) for image in images]).to(device)
        y = x.maxpool(x.relu(x.bn1(x.conv1(image_batch))))
        layer4 = x.layer4(x.layer3(x.layer2(x.layer1(y))))
        if layer4.shape[1:] != (512, 7, 7):
            raise ValueError(f"Unexpected R3M layer4 shape: {tuple(layer4.shape)}")
        spatial = F.adaptive_avg_pool2d(layer4, (3, 3)).flatten(2).transpose(1, 2)
        return F.normalize(spatial, dim=-1), F.normalize(layer4.mean((-2, -1)), dim=-1)


def merge_annotations(rows):
    groups = []
    for language, duration in rows:
        if type(duration) is not int or duration <= 0:
            raise ValueError(f"Invalid annotation duration: {duration!r}")
        if groups and groups[-1][0] == language:
            groups[-1][1] += duration
        else:
            groups.append([language, duration])
    return groups


class CoverVisualVerifier:
    """Offline evidence for a static camera and the official Cover Blocks geometry."""

    SITES = ("left", "middle", "right")
    FRONT_Y, PARK_Y = 175, 127
    X = (91, 173, 254)

    def __init__(self, initial):
        if initial.shape != (240, 320, 3):
            raise ValueError(f"Unexpected Cover Blocks camera shape: {initial.shape}")
        self.colors = []
        self.blocks = []
        self.lids = []
        for x in self.X:
            blueprints = self.block_pixels(self.patch(initial, x, self.FRONT_Y, 13, 13))
            index = int(blueprints.argmax())
            self.colors.append(index)
            self.blocks.append(int(blueprints[index]))
            self.lids.append(self.tan_pixels(self.patch(initial, x, self.PARK_Y, 21, 18)))
        if sorted(self.colors) != [0, 1, 2] or min(self.blocks) < 60 or min(self.lids) < 250:
            raise ValueError("Expected separated colored cubes and lid templates in the initial frame")

    @staticmethod
    def patch(rgb, x, y, width, height):
        return rgb[y-height:y+height+1, x-width:x+width+1].astype(np.int16)

    @staticmethod
    def block_pixels(patch):
        return np.array([np.count_nonzero((patch[..., c] >= 100)
                         & (patch[..., c] - np.maximum(patch[..., (c+1)%3], patch[..., (c+2)%3]) >= 50))
                         for c in range(3)])

    @staticmethod
    def tan_pixels(patch):
        r, g, b = np.moveaxis(patch, -1, 0)
        return int(np.count_nonzero((r-g >= 5) & (g-b >= 7) & (r-b >= 20)
                                    & (r-b <= 95) & (r >= 70) & (r <= 240)))

    def check(self, image, covered, operated_site, state):
        results = []
        for i, x in enumerate(self.X):
            front = self.patch(image, x, self.FRONT_Y, 21, 18)
            back = self.patch(image, x, self.PARK_Y, 21, 18)
            visible = int(self.block_pixels(self.patch(image, x, self.FRONT_Y, 13, 13))[self.colors[i]])
            at_front, parked = self.tan_pixels(front), self.tan_pixels(back)
            if covered[i]:
                good = visible <= self.blocks[i] * .12 and at_front >= self.lids[i] * .4
            else:
                good = visible >= self.blocks[i] * .40 and parked >= self.lids[i] * .40
            results.append({"cube_pixels": visible, "front_lid_pixels": at_front,
                            "park_lid_pixels": parked, "matches": bool(good)})
        gripper = float(state[6 if operated_site == 0 else 13])
        return all(x["matches"] for x in results) and gripper >= .85, results, gripper


def save_sheet(thumb_dir, episode_id, panels):
    rows = []
    for spans in panels:
        cols = []
        for caption, rgb in spans:
            thumb = (cv2.resize(rgb, (160, 120), interpolation=cv2.INTER_AREA) if rgb is not None
                     else np.zeros((120, 160, 3), np.uint8))
            thumb = cv2.copyMakeBorder(thumb.copy(), 18, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(thumb, caption, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, .32, (255, 255, 255), 1)
            cols.append(thumb)
        rows.append(np.hstack(cols))
    path = thumb_dir / f"episode_{episode_id:03d}.jpg"
    if not cv2.imwrite(str(path), np.vstack(rows)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 83]):
        raise RuntimeError(f"Could not save {path}")


def collect(args, dataset, decode):
    if dataset.manifest["task"] != "cover_blocks" or [ep["id"] for ep in dataset.episodes] != list(range(DEMO_COUNT)):
        raise ValueError("Expected the 50 official Cover Blocks demonstrations")
    thumb_dir = args.output / "keyframes"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    all_events, keyframes, audit = [], {}, []
    camera_calibration = None
    with ZipFile(args.archive) as archive:
        member, = (name for name in archive.namelist() if name.endswith("/language_annotation.json"))
        annotation = json.loads(archive.read(member))
        if set(annotation) != {f"episode_{ep['id']}" for ep in dataset.episodes}:
            raise ValueError("Raw and prepared demonstration IDs differ")
        for ep in dataset.episodes:
            eid, frames = ep["id"], ep["steps"]
            groups = merge_annotations(annotation[f"episode_{eid}"])
            durations = [length for _, length in groups]
            if durations != [phase["frames"] for phase in ep["official_cover_phases"]]:
                raise ValueError(f"Episode {eid} has ambiguous operation intervals")
            source_events = events_from_durations(eid, durations, frames, member)
            sha_input = io.BytesIO(archive.read(ep["archive_member"]))
            with h5py.File(sha_input, "r") as h5:
                images, state, actions, source = _read_expert(h5)
                prepared = dataset._load(eid, "rgb")
                if (source["layout"] != "XPolicyLab" or len(images) != len(prepared) or len(images) != frames
                        or not np.array_equal(state, dataset._load(eid, "state"))
                        or not np.array_equal(actions, dataset._load(eid, "action"))
                        or not np.array_equal(actions[:-1], state[1:])):
                    raise ValueError(f"Prepared RGB/state/action alignment failed for episode {eid}")
                camera = h5["vision/cam_head"]
                calibration = tuple(np.asarray(camera[n]) for n in ("intrinsic_matrix", "extrinsics_matrix"))
                if camera_calibration is None:
                    camera_calibration = tuple(a[0] for a in calibration)
                if not all(np.allclose(actual, reference, atol=1e-5)
                           for actual, reference in zip(calibration, camera_calibration)):
                    raise ValueError(f"Cover verifier requires fixed calibrated camera in episode {eid}")
                cache = {}
                def rgb(index):
                    if index not in cache:
                        cache[index] = decode(images[index])
                    return cache[index]

                verifier = CoverVisualVerifier(rgb(0))
                covered = [False] * 3
                panels = []
                revealed_colors = []
                for event, (text, _), phase in zip(source_events, groups, ep["official_cover_phases"]):
                    verb, site = phase["operation"]
                    slot = verifier.SITES.index(site)
                    covered[slot] = verb == "Cover"
                    if verb == "Open":
                        revealed_colors.append(verifier.colors[slot])
                    start, end = event["start_frame"], event["end_frame"]
                    middle, default = start + (end-start)//2, end-1
                    for i in (start, middle, default):
                        if not np.array_equal(preprocess_rgb(rgb(i)), prepared[i]):
                            raise ValueError(f"RGB index mismatch in episode {eid} frame {i}")
                    goal = None
                    detail = None
                    for t in range(default, max(middle, start+2)-1, -1):
                        measurements = [verifier.check(rgb(i), covered, slot, state[i]) for i in range(t-2,t+1)]
                        if all(item[0] for item in measurements):
                            goal = t
                            detail = {"sites": measurements[-1][1], "gripper_open": measurements[-1][2]}
                            break
                    valid_end = min(middle, goal) if goal is not None else start
                    index = len(all_events)
                    metadata = {"episode_id":eid,"event_id":event["event_id"],"start_frame":start,"end_frame":end,
                                "goal_frame_index":goal,"goal_feature_index":index if goal is not None else None,
                                "valid_mask":{"start_frame":start,"end_frame":valid_end},
                                "annotation_source":member}
                    all_events.append(metadata)
                    if goal is not None:
                        if not np.array_equal(preprocess_rgb(rgb(goal)), prepared[goal]):
                            raise ValueError(f"Selected RGB index mismatch in episode {eid} frame {goal}")
                        keyframes[index] = rgb(goal)
                    reason = ("end_matches_completed_state" if goal == default else
                              "selected_last_stable_visible_completion" if goal is not None else
                              "no_stable_visible_completion_inside_event")
                    audit.append({**metadata,"candidate_frame":default,"operation":text,
                                  "diagnostic_operation":f"{verb} {site}",
                                  "diagnostic_state":"".join(str(int(c)) for c in covered),
                                  "reason":reason,"evidence":detail,"image_summary":f"keyframes/episode_{eid:03d}.jpg"})
                    t = goal if goal is not None else default
                    panels.append([(f"start {start}",rgb(start)),(f"mid {middle}",rgb(middle)),
                                   (f"candidate {default}",rgb(default)),
                                   (f"goal prev {t-1}",rgb(t-1)),(f"goal {t} {goal is not None}",rgb(t)),
                                   (f"goal next {t+1}",rgb(t+1) if t+1<frames else None)])
                if revealed_colors != [0,1,2]:
                    raise ValueError(f"Expert open order and initial RGB disagree for episode {eid}: {revealed_colors}")
                save_sheet(thumb_dir,eid,panels)
            if (eid+1)%10==0:
                print(f"Audited {eid+1} of {DEMO_COUNT} demonstrations",flush=True)
    return all_events, keyframes, audit



def describe(values):
    return {"n": int(len(values)), "mean": float(np.mean(values)),
            "quantiles_05_50_95": np.quantile(values, (.05, .5, .95)).tolist()}


def retrieval(features, audit, label, indices):
    labels = np.asarray([row[label] for row in audit])[indices]
    episodes = np.asarray([row["episode_id"] for row in audit])[indices]
    other = episodes[:, None] != episodes[None, :]
    same = labels[:, None] == labels[None, :]
    vectors = features[indices].reshape(len(indices), -1)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarities = vectors @ vectors.T
    scores = np.where(other, similarities, -np.inf)
    nearest = scores.argmax(axis=1)
    closest_positive = scores.copy()
    closest_positive[~same] = -np.inf
    nearest_positive = closest_positive.argmax(axis=1)
    positive = similarities[other & same]
    negative = similarities[other & ~same]
    top1 = labels[nearest] == labels
    unique = np.unique(labels)
    per_class = {str(c): float(top1[labels == c].mean()) for c in unique}
    sorted_negative = np.sort(negative)
    auroc = (np.searchsorted(sorted_negative, positive, side="left")
             + np.searchsorted(sorted_negative, positive, side="right")).mean() / (2*len(negative))
    representatives = []
    for success in (True, False):
        for i in np.flatnonzero(top1 == success)[:3]:
            j = nearest[i]
            pa = nearest_positive[i]
            representatives.append({"correct": bool(success), "query_event": int(indices[i]),
                                    "nearest_event": int(indices[j]), "same_class_example": int(indices[pa]),
                                    "query_label": labels[i].item(), "nearest_label": labels[j].item(),
                                    "similarity": float(scores[i,j])})
    return {"top1": float(top1.mean()), "balanced_top1": float(np.mean(list(per_class.values()))),
            "balanced_chance": 1 / len(unique), "per_class_top1": per_class,
            "same_class_similarity": describe(positive), "different_class_similarity": describe(negative),
            "pair_auroc": float(auroc), "examples": representatives}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


@torch.inference_mode()
def main():
    from XPolicyLab.utils.process_data import decode_image_bit
    from goal_targets import encode_keyframes
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "data", "r3m-weights", "global-weights", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    dataset = EpisodeDataset(args.data)
    events, frames, audit = collect(args, dataset, decode_image_bit)
    if not frames:
        raise ValueError("No visually verified event completion")
    ids = np.array(list(frames), dtype=np.int64)
    teacher = R3MSpatial(args.r3m_weights).to(args.device)
    spatial = np.zeros((len(events), *SHAPE), np.float32)
    global_r3m = np.zeros((len(events), 512), np.float32)
    images = list(frames.values())
    for start in range(0, len(ids), 32):
        stop = start + 32
        s, g = teacher(images[start:stop])
        spatial[ids[start:stop]] = s.cpu().numpy()
        global_r3m[ids[start:stop]] = g.cpu().numpy()
    old_features, old_teacher = encode_keyframes(images, args.global_weights, args.device)
    legacy = np.zeros_like(global_r3m)
    legacy[ids] = old_features
    r3m = {"teacher": "official author release R3M-ResNet18", "repository": R3M_REPOSITORY,
           "revision": R3M_REVISION,
           "weight_sha256": hashlib.sha256(args.r3m_weights.read_bytes()).hexdigest(),
           "preprocess": {"channel_order":"RGB","input_dtype":"uint8","input_range":[0,255],
                          "resize_short_edge":256,"center_crop":[224,224],
                          "interpolation":"PIL bilinear","scale":"divide by 255 exactly once",
                          "normalize_mean":MEAN,"normalize_std":STD},
           "layer":"layer4","layer_shape":[512,7,7],"pool":"adaptive avg-pool 3x3, flatten row-major",
           "target_shape":SHAPE,"l2_normalization":"per 512-dimensional spatial location"}
    np.save(args.output / "goal_features.npy", spatial, allow_pickle=False)
    write_json(args.output / "events.json", {"schema_version":2,"target_shape":SHAPE,
               "teacher":r3m,"task":dataset.manifest["task"],"events":events})
    write_json(args.output / "keyframe_audit.json", audit)
    targets = GoalTargets(args.output, dataset.episodes, SHAPE)
    stats = {"teacher":r3m,"comparison_teacher":old_teacher,"episodes":len(dataset.episodes),
             "candidate_events":len(events),"valid_events":len(ids),
             "default_frames":sum(e["reason"]=="end_matches_completed_state" for e in audit),
             "adjusted_frames":sum(e["reason"]=="selected_last_stable_visible_completion" for e in audit),
             "invalid_events":len(events)-len(ids),
             "raw_frames":sum(ep["steps"]+1 for ep in dataset.episodes),
             "paired_frames":sum(ep["steps"] for ep in dataset.episodes),
             "valid_training_anchors":sum(int((targets.anchor_index[e["id"]]
                [:dataset._anchor_count(e["steps"],16,8)]>=0).sum()) for e in dataset.episodes)}
    write_json(args.output / "statistics.json",stats)
    opened = np.array([i for i in ids if audit[i]["diagnostic_operation"].startswith("Open ")])
    quality = {"protocol":"Leave out the entire query demonstration from its neighbor gallery; same verified goal frames across teachers.",
               "threshold":"balanced nearest-neighbor accuracy for completed-state pattern and uncover-site must exceed their balanced chance by 10 percentage points",
               "measures":{}}
    for name, features in (("r3m_spatial",spatial),("legacy_imagenet_global",legacy),("r3m_global",global_r3m)):
        quality["measures"][name] = {"state":retrieval(features,audit,"diagnostic_state",ids),
                                    "open_site":retrieval(features,audit,"diagnostic_operation",opened)}
    quality["pass"] = all(x["balanced_top1"] >= x["balanced_chance"]+.1
                          for x in quality["measures"]["r3m_spatial"].values())
    write_json(args.output / "feature_quality.json",quality)
    print(json.dumps({"valid":len(ids),"total":len(events),"adjusted":stats["adjusted_frames"],
           "feature_quality_pass":quality["pass"]}),flush=True)


if __name__ == "__main__":
    main()
