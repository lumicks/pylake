def classify_track(red, blue, timing_window=6):
    """This function classifies the tracks into groups based on the order in which
    the proteins bind"""

    #  3 blue_start < red_start  blue_end < red_end
    #  4 blue_start < red_start  blue_end = red_end
    #  5 blue_start < red_start  blue_end > red_end

    #  6 blue_start = red_start  blue_end < red_end
    #  7 blue_start = red_start  blue_end = red_end
    #  8 blue_start = red_start  blue_end > red_end

    #  9 red_start < blue_start  blue_end < red_end
    # 10 red_start < blue_start  blue_end = red_end
    # 11 red_start < blue_start  blue_end > red_end

    def classify(time, ref_time):
        if ref_time - time > timing_window:
            return 0
        elif time - ref_time > timing_window:
            return 2
        else:
            return 1

    return (
        3
        + 3 * classify(blue.time_idx[0], red.time_idx[0])
        + classify(blue.time_idx[-1], red.time_idx[-1])
    )
