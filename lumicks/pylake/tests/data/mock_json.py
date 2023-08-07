import json

force_feedback_dict = {
    "settings_at_start": {
        "enabled_status": 0,
        "kp": 0.0,
        "ki": 0.0,
        "kd": 0.0,
        "setpoint": 0.0,
        "lower_output_limit": 0.0,
        "upper_output_limit": 2.0,
    },
    "amplified_input_factor": 10.0,
    "amplified_input_offset": 0.0,
    "start_time": {"time_since_epoch": {"count": 1638883620243399988}},
}


def mock_json(data_dict):
    """Mocks a json as we receive them from BL"""
    return json.JSONEncoder().encode({"value0": data_dict})


def mock_force_feedback_json():
    return json.JSONEncoder().encode(mock_json(force_feedback_dict))
