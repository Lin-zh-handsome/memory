"""Behavioral checks for the RMBench M2/M3 causal policy contract."""
import argparse
import math

import numpy as np
import torch
from torch import nn

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

from models import (
    HISTORY_DIM, MEMORY_CAPACITY, Event, OrderedEvents, build_policy, mamba_recurrence,
)


def close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=2e-5)


def assert_gradient(param):
    assert param.grad is not None
    assert torch.isfinite(param.grad).all()
    assert torch.count_nonzero(param.grad)


def check_mamba(device):
    core = Mamba(d_model=256, d_state=16, d_conv=4, expand=2).to(device)
    tokens = torch.randn(2, 43, 256, device=device)
    with torch.no_grad():
        full, terminal = mamba_recurrence(core, tokens)
        if device.type == "cuda":
            close(full, core(tokens))
        else:
            # The upstream kernel is CUDA-only. The upstream reference scan
            # still gives an independent causal full-sequence baseline on CPU.
            x, z = core.in_proj(tokens).transpose(1, 2).chunk(2, dim=1)
            x = core.act(core.conv1d(x)[..., :tokens.shape[1]])
            dt, b, c = core.x_proj(x.transpose(1, 2)).split(
                (core.dt_rank, core.d_state, core.d_state), dim=-1,
            )
            ref = selective_scan_ref(
                x, nn.functional.linear(dt, core.dt_proj.weight).transpose(1, 2),
                -core.A_log.exp(), b.transpose(1, 2), c.transpose(1, 2),
                core.D, z=z, delta_bias=core.dt_proj.bias, delta_softplus=True,
            )
            close(full, core.out_proj(ref.transpose(1, 2)))

        cache = None
        parts = []
        for segment in (tokens[:, :7], tokens[:, 7:23], tokens[:, 23:]):
            part, cache = mamba_recurrence(core, segment, cache)
            parts.append(part)
        close(torch.cat(parts, dim=1), full)
        for actual, expected in zip(cache, terminal):
            close(actual, expected)

        cache = None
        parts = []
        for step in tokens.split(1, dim=1):
            part, cache = mamba_recurrence(core, step, cache)
            parts.append(part)
        close(torch.cat(parts, dim=1), full)
        for actual, expected in zip(cache, terminal):
            close(actual, expected)

        if device.type == "cuda":
            conv, ssm = core.allocate_inference_cache(tokens.shape[0], 1)
            installed = []
            for step in tokens.split(1, dim=1):
                part, conv, ssm = core.step(step, conv, ssm)
                installed.append(part)
            close(torch.cat(installed, dim=1), full)
            close(conv, terminal[0])
            close(ssm, terminal[1])

        changed = tokens.clone()
        changed[:, :9].neg_()
        changed_history, _ = mamba_recurrence(core, changed)
        # At t=24 the last 16 input tokens agree, but the SSM still carries
        # earlier history. The same mechanism persists across longer episodes.
        assert (full[:, 24] - changed_history[:, 24]).abs().max() > 1e-7
        _, (conv, ssm) = mamba_recurrence(core, tokens[:, :23])
        expected, _ = mamba_recurrence(core, tokens[:, 23:24], (conv, ssm))
        for broken in ((conv * 0, ssm), (conv, ssm * 0)):
            reset, _ = mamba_recurrence(core, tokens[:, 23:24], broken)
            assert (expected - reset).abs().max() > 1e-6

    first = tokens[:, :7].clone().requires_grad_()
    later = tokens[:, 7:23].clone().requires_grad_()
    _, cache = mamba_recurrence(core, first)
    carry = tuple(tensor.detach().requires_grad_() for tensor in cache)
    result, terminal = mamba_recurrence(core, later, carry)
    (result.square().mean() + sum(state.square().mean() for state in terminal)).backward()
    assert first.grad is None
    assert_gradient(later)
    for value, previous in zip(carry, cache):
        close(value, previous)
        assert_gradient(value)
    for param in (core.in_proj.weight, core.conv1d.weight, core.A_log, core.dt_proj.weight):
        assert_gradient(param)


def event(time, angles, value=None):
    value = time if value is None else value
    def scores(angle):
        radians = math.radians(angle)
        return np.array([math.cos(radians), math.sin(radians)], dtype=np.float32)
    return Event(
        torch.full((HISTORY_DIM,), float(value)),
        torch.full((HISTORY_DIM,), -float(value)),
        scores(angles[0]), scores(angles[1]), time, time == 0,
    )


def check_event_bank():
    bank = OrderedEvents((0.97, 0.8))
    reference = torch.zeros(HISTORY_DIM)
    assert not torch.count_nonzero(bank.read(0, reference, 1))
    assert bank.write(event(0, (0, 0), value=10))[0]
    assert not bank.write(event(1, (0, 10)))[0]
    assert bank.write(event(2, (0, 90)))[0]
    assert bank.write(event(3, (60, 90)))[0]
    rows = bank.read(4, reference, 1).reshape(MEMORY_CAPACITY, -1)
    assert [record.time for record in bank.records] == [0, 2, 3]
    close(rows[:3, 0], torch.tensor([10., 2., 3.]))
    close(rows[:3, HISTORY_DIM], torch.tensor([-10., -2., -3.]))
    close(rows[:3, -2], torch.log1p(torch.tensor([4., 2., 1.])))
    close(rows[:, -1], torch.tensor([1., 1., 1., 0., 0., 0., 0., 0.]))
    assert not torch.count_nonzero(rows[3:])

    # Pair (1,2) has larger arithmetic mean; the prescribed minimum selects (2,3).
    angles = [(0, 0), (0, 0), (0, 60), (45, 105), (135, 195), (225, 285), (315, 15), (45, 105)]
    bank = OrderedEvents((0.97, 0.8), [event(t, pair) for t, pair in enumerate(angles)])
    written, removed, _ = bank.write(event(8, (135, 105)))
    assert written and removed == 2
    assert len(bank.records) == MEMORY_CAPACITY and bank.records[0].protected
    assert [record.time for record in bank.records] == [0, 1, 3, 4, 5, 6, 7, 8]
    bank = OrderedEvents((0.97, 0.8), [event(t, (0, 0)) for t in range(MEMORY_CAPACITY)])
    assert bank.write(event(8, (90, 90)))[1] == 1


def sample_policy(variant, device):
    cfg = {
        "vision": {"crop": None, "imagenet_norm": False},
        "ddpm": {"num_train_timesteps": 3, "prediction_type": "epsilon"},
        "horizon": 16, "n_action_steps": 8, "num_inference_steps": 3,
        "event_thresholds": {"visual": 0.9999, "history": 0.9998},
        "unet": {"down_dims": [32, 64, 128], "diffusion_step_embed_dim": 16},
    }
    shape = {
        "action": {"shape": [14]},
        "obs": {
            "head_camera": {"shape": [3, 32, 32], "type": "rgb"},
            "agent_pos": {"shape": [14], "type": "low_dim"},
        },
    }
    policy = build_policy(variant, shape, cfg)
    normalizer = LinearNormalizer()
    normalizer.fit({"agent_pos": torch.randn(30, 14), "action": torch.randn(30, 14)})
    normalizer["head_camera"] = get_image_range_normalizer()
    policy.set_normalizer(normalizer)
    return policy.to(device)


def check_policy(device, variants=("M2", "M3")):
    obs = {
        "head_camera": torch.rand(25, 3, 32, 32, device=device),
        "agent_pos": torch.rand(25, 14, device=device),
    }
    def frames(start, end):
        return {key: value[start:end] for key, value in obs.items()}

    for variant in variants:
        policy = sample_policy(variant, device).eval()
        with torch.no_grad():
            full, full_history, full_state = policy.segment_condition(obs)
            assert full.shape[-1] == 2 * policy.obs_feature_dim + HISTORY_DIM * (1 if variant == "M2" else 2)
            first, _, carry = policy.segment_condition(frames(0, 12))
            carry = carry.detach()
            second, _, carry = policy.segment_condition(frames(12, 25), carry)
            close(torch.cat((first, second)), full)
            close(carry.conv, full_state.conv)
            close(carry.ssm, full_state.ssm)

            policy.reset_memory(["a", "b"])
            captures = []
            for index in range(25):
                policy.update_observation(
                    {key: torch.stack((value[index], value[24 - index])) for key, value in obs.items()},
                    ["a", "b"],
                )
                captures.append(policy.get_history_feature(["a"])[0])
                close(policy._online["a"].condition, full[index])
                if index in (0, 8, 16, 24):
                    before = policy.memory_diagnostics(["a", "b"])
                    hook = policy.model.register_forward_pre_hook(
                        lambda module, args, kwargs: close(kwargs["global_cond"], full[index:index + 1]),
                        with_kwargs=True,
                    )
                    predicted = policy.predict_stream(["a"])
                    hook.remove()
                    close(predicted["action"], predicted["action_pred"][:, 1:9])
                    assert before == policy.memory_diagnostics(["a", "b"])
            close(torch.stack(captures), full_history)
            assert policy.memory_diagnostics(["a"])[0]["observations"] == 25
            if variant == "M3":
                stats = policy.memory_diagnostics(["a"])[0]
                assert stats["initial_retained"] and stats["replacement_count"] > 0
                assert max(stats["capacities"]) == 8
                assert not torch.count_nonzero(full[0, -HISTORY_DIM:])
                assert torch.count_nonzero(full[1, -HISTORY_DIM:])
            else:
                assert not hasattr(policy, "memory_projection")

            other = policy.get_history_feature(["b"]).clone()
            other_stats = policy.memory_diagnostics(["b"])
            policy.reset_memory(["a"])
            assert policy.memory_diagnostics(["a"])[0]["observations"] == 0
            assert policy.memory_diagnostics(["a"])[0]["final_event_times"] == []
            close(policy._online["a"].conv, torch.zeros_like(policy._online["a"].conv))
            close(policy._online["a"].ssm, torch.zeros_like(policy._online["a"].ssm))
            close(policy.get_history_feature(["b"]), other)
            assert policy.memory_diagnostics(["b"]) == other_stats
            policy.update_observation(frames(0, 1), ["a"])
            close(policy.get_history_feature(["a"])[0], full_history[0])

        policy.train()
        batch = {
            "obs": frames(0, 6),
            "action": torch.randn(6, 16, 14, device=device),
            "past_action": torch.randn(6, 8, 14, device=device),
            "past_mask": torch.arange(6, device=device)[:, None] > torch.arange(8, device=device)[None, :],
        }
        loss, state = policy.compute_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        for param in (policy.history_mamba.in_proj.weight, policy.past_head[0].weight):
            assert_gradient(param)
        if variant == "M3":
            for param in (policy.memory_projection.weight, policy.event_visual_projection.weight):
                assert_gradient(param)
        assert state.conv.grad_fn is not None and state.ssm.grad_fn is not None
        detached = state.detach()
        assert detached.conv.grad_fn is None and detached.ssm.grad_fn is None
        close(detached.conv, state.conv)
        close(detached.ssm, state.ssm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=torch.device, required=True)
    args = parser.parse_args()
    torch.manual_seed(18)
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    check_mamba(args.device)
    print("PASS Mamba1 causal windows, both caches, upstream parity and TBPTT gradients", flush=True)
    check_event_bank()
    print("PASS event novelty, protected ordered memory, min redundancy and time/mask", flush=True)
    check_policy(args.device)
    print("PASS M2/M3 online lifecycle, eight-step chunks and DDPM memory gradients", flush=True)

