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
    elif task == "battery_try":
        state.update(batteries=[pose_record(env.battery1), pose_record(env.battery2)],
                     slot=[pose_record(env.battery_slot)],
                     target_combination=[int(bit) for bit in env.correct_combination])
    elif task == "observe_and_pickup":
        state.update(objects=[pose_record(actor) for actor in env.object],
                     target_index=int(env.target_object_idx),
                     object_models=list(zip(env.object_modelnames, map(int, env.object_ids))))
    elif task == "rearrange_blocks":
        state.update(blocks=[pose_record(env.block1), pose_record(env.block2)],
                     target_positions=[env.target_pose1.p.tolist(), env.target_pose2.p.tolist()])
    elif task == "swap_T":
        state.update(blocks=[pose_record(env.T_block1), pose_record(env.T_block2)],
                     target_positions=[np.asarray(env.target_pose2).tolist(), np.asarray(env.target_pose1).tolist()])
    elif task == "blocks_ranking_try":
        state.update(blocks=[pose_record(actor) for actor in (env.block1, env.block2, env.block3)])
    elif task == "press_button":
        state.update(target_counts=[int(env.card_id_1), int(env.card_id_2)])
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
        return {**state, "stage": int(env.current_state_pointer), "invalid": bool(env.fail_flag),
                "cover_pattern": [bool(env.check_if_cover(i)) for i in range(3)]}
    if task == "battery_try":
        placed = [bool(env.check_battery_in_slot(actor, target)) for actor, target in
                  ((env.battery1, env.target_pose1_p), (env.battery2, env.target_pose2_p))]
        combination = [int(bit) for bit in env.get_curr_combination()] if all(placed) else None
        return {**state, "stage": 3 if env.eval_success else sum(placed), "installed": placed,
                "combination": combination, "gauge_on": bool(env.check_dashboard_on())}
    if task == "observe_and_pickup":
        contacts = {actor.actor.per_scene_id: index for index, actor in enumerate(env.object)}
        fingers = [set() for _ in env.object]
        for contact in env.scene.get_contacts():
            first, second = [body.entity for body in contact.bodies]
            for finger, obj in ((first, second), (second, first)):
                index = contacts.get(obj.per_scene_id)
                if index is not None and finger.name in env.robot.gripper_name:
                    fingers[index].add(finger.name)
        grasped = [index for index, names in enumerate(fingers) if len(names) >= 2]
        return {**state, "stage": int(bool(env.eval_success)), "invalid": bool(env.fail_flag),
                "success_checks": int(env.get_obs_cnt), "wall_visible": int(env.get_obs_cnt) >= 20,
                "target_index": int(env.target_object_idx), "grasped_objects": grasped,
                "target_height_m": float(env.object[env.target_object_idx].get_pose().p[2])}
    if task == "rearrange_blocks":
        return {**state, "stage": int(env.stage_id), "button_presses": int(env.press_cnt),
                "invalid": int(env.press_cnt) > 1,
                "block_positions": [actor.get_pose().p.tolist() for actor in (env.block1, env.block2)]}
    if task == "swap_T":
        actors = (env.T_block1, env.T_block2)
        goals = (env.target_pose2, env.target_pose1)
        errors = [float(np.linalg.norm(actor.get_pose().p[:2] - goal[:2]))
                  for actor, goal in zip(actors, goals)]
        return {**state, "stage": int(bool(env.eval_success)), "xy_errors_m": errors,
                "blocks": [pose_record(actor) for actor in actors]}
    if task == "blocks_ranking_try":
        positions = [actor.get_pose().p.tolist() for actor in (env.block1, env.block2, env.block3)]
        order = np.argsort([position[0] for position in positions]).tolist()
        return {**state, "stage": int(bool(env.eval_success)), "block_order": order,
                "button_presses": int(env.press_cnt), "block_positions": positions}
    if task == "press_button":
        return {**state, "stage": int(bool(env.eval_success)),
                "button_counts": [int(env.press_cnt_1), int(env.press_cnt_2), int(env.press_cnt_check_button)],
                "target_counts": [int(env.card_id_1), int(env.card_id_2)],
                "check_button_pressed": bool(env.press_flag_check_button)}
    if task != "put_back_block":
        return {**state, "stage": int(bool(env.eval_success))}
    position = np.asarray(env.block.get_pose().p)
    error = np.linalg.norm(position[:2] - np.asarray(env.target_pose)[:2])
    return {**state, "stage": int(env.stage_id), "button_presses": int(env.press_cnt),
            "xy_return_error_m": float(error)}


def transition_key(state):
    return (state["stage"], state.get("invalid"), state.get("combination"), state.get("gauge_on"),
            state.get("wall_visible"), state.get("grasped_objects"), state.get("button_presses"),
            state.get("block_order"), state.get("button_counts"))


def outcome_details(task, trace, env):
    if task == "observe_and_pickup":
        first = next((row for row in trace if row.get("grasped_objects")), None)
        category = None
        if not env.eval_success:
            if env.fail_flag:
                category = "early_motion"
            elif first is None:
                category = "no_two_finger_contact"
            elif first["grasped_objects"] != [int(env.target_object_idx)]:
                category = "first_contact_wrong_object"
            else:
                category = "target_contact_without_success"
        return {"failure_category": category, "first_two_finger_contact": first,
                "first_contact_correct": bool(first and first["grasped_objects"] == [int(env.target_object_idx)])}
    if task not in ("cover_blocks", "battery_try"):
        return {}
    final = report_state(task, env)
    failure = None
    if not env.eval_success:
        if task == "cover_blocks":
            stage = max(row["stage"] for row in trace)
            failure = ("invalid_cover_order" if stage < 3 else "invalid_uncover_order") if final["invalid"] else (
                "cover_phase_timeout" if stage < 3 else "uncover_phase_timeout"
            )
        else:
            failure = "combination_timeout" if any(row["combination"] is not None for row in trace) else "placement_timeout"
    details = {"failure_category": failure, "timed_out": bool(not env.eval_success and env.take_action_cnt >= env.step_lim)}
    if task == "battery_try":
        attempts = []
        last = None
        for row in trace:
            combination = row["combination"]
            if combination is not None and combination != last:
                attempts.append(combination)
            last = combination
        unique = {tuple(bits) for bits in attempts}
        details["battery_attempts"] = {
            "combinations": attempts, "attempt_count": len(attempts), "distinct_count": len(unique),
            "repeated_count": len(attempts) - len(unique), "gauge_activated": any(row["gauge_on"] for row in trace),
            "successful_feedback": bool(env.eval_success and final["gauge_on"]),
        }
    return details


def track_controller_fallbacks(task, env):
    """Count planner failures without changing the official controller behavior."""
    if task not in ("cover_blocks", "battery_try"):
        return None
    counts = {"left": 0, "right": 0}

    def instrument(planner, side):
        original = planner.TOPP

        def tracked(*args, **kwargs):
            try:
                result = original(*args, **kwargs)
            except Exception:
                counts[side] += 1
                raise
            if result[1].shape[0] == 0:
                counts[side] += 1
            return result

        planner.TOPP = tracked

    instrument(env.robot.left_mplib_planner, "left")
    instrument(env.robot.right_mplib_planner, "right")
    return counts
