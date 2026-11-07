"""Official-task state used only for reset auditing and trajectory logs."""
import numpy as np


def pose_record(actor):
    pose = actor.get_pose()
    return {"p": pose.p.tolist(), "q": pose.q.tolist()}


def initial_state(task, env, qpos):
    state = {"qpos": qpos.tolist()}
    if task == "swap_blocks":
        state.update(
            baskets=[pose_record(basket) for basket in env.baskets],
            blocks=[pose_record(env.block1), pose_record(env.block2)],
            occupied_baskets=[int(env.with_block_basket_idx_1), int(env.with_block_basket_idx_2)],
            empty_basket=int(env.empty_basket_idx),
        )
    elif task == "cover_blocks":
        state.update(blocks=[pose_record(block) for block in env.blocks],
                     covers=[pose_record(cover) for cover in env.covers],
                     uncover_order=list(env.open_lst))
    return state


def swap_state(env):
    """Describe observed progress. The official simulator decides success."""
    baskets = [basket.get_pose().p for basket in env.baskets]
    locations = []
    for block in (env.block1, env.block2):
        position = block.get_pose().p
        matches = [i for i, tray in enumerate(baskets)
                   if abs(tray[0] - position[0]) < 0.04
                   and abs(tray[1] - position[1]) < 0.06 and position[2] < 0.765]
        locations.append(matches[0] if matches else None)
    original_first, original_second = env.with_block_basket_idx_1, env.with_block_basket_idx_2
    expected = ([env.empty_basket_idx, original_second], [env.empty_basket_idx, original_first],
                [original_second, original_first])
    stage = next((index for index, pair in enumerate(expected, 1) if locations == pair), 0)
    if env.eval_success:
        stage = 4
    return {"stage": stage, "block_baskets": locations,
            "button_presses": int(env.press_cnt), "invalid": not bool(env.flag)}


def report_state(task, env):
    state = {"step": int(env.take_action_cnt)}
    if task == "swap_blocks":
        return {**state, **swap_state(env)}
    if task == "cover_blocks":
        return {**state, "stage": int(env.current_state_pointer), "invalid": bool(env.fail_flag)}
    position = np.asarray(env.block.get_pose().p)
    error = np.linalg.norm(position[:2] - np.asarray(env.target_pose)[:2])
    return {**state, "stage": int(env.stage_id), "button_presses": int(env.press_cnt),
            "xy_return_error_m": float(error)}
