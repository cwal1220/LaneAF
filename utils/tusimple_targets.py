AREA_CHANNEL_NAMES = ("ego_lanes_left", "ego_lanes", "ego_lanes_right")
LINE_CHANNEL_NAMES = ("ego_lines_left", "ego_lines", "ego_lines_right")

TUSIMPLE_HEADS = {
    "lane_hm": len(AREA_CHANNEL_NAMES),
    "line_hm": len(LINE_CHANNEL_NAMES),
    "vaf": 2,
    "haf": 1,
    "line_vaf": len(LINE_CHANNEL_NAMES) * 2,
    "line_haf": len(LINE_CHANNEL_NAMES),
}


def target_groups():
    return (
        ("area", "lane_hm", AREA_CHANNEL_NAMES),
        ("line", "line_hm", LINE_CHANNEL_NAMES),
    )
