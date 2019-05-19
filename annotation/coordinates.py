
def coord_convert(x, y, window_location):
    scale_to_original, left_top_x, left_top_y, width, height = window_location
    new_x = (x+left_top_x) / scale_to_original
    new_y = (y+left_top_y) / scale_to_original
    return new_x, new_y
