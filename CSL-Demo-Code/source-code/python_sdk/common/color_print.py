"""
A module to help with color printing and bold printing
"""


class Color:
    """
    Colors to be passed in as arguments to color_print and bold_print
    """
    # foreground
    default = '\033[39m'
    black = '\033[30m'
    red = '\033[31m'
    green = '\033[32m'
    yellow = '\033[33m'
    blue = '\033[34m'
    magenta = '\033[35m'
    cyan = '\033[36m'
    light_gray = '\033[37m'
    dark_gray = '\033[90m'
    light_red = '\033[91m'
    light_green = '\033[92m'
    light_yellow = '\033[93m'
    light_blue = '\033[94m'
    light_magenta = '\033[95m'
    light_cyan = '\033[96m'
    white = '\033[97m'
    bold = '\033[1m'

    # backgroud
    default_background = '\033[49m'
    black_background = '\033[40m'
    red_background = '\033[41m'
    green_background = '\033[42m'
    yellow_background = '\033[43m'
    blue_background = '\033[44m'
    magenta_background = '\033[45m'
    cyan_background = '\033[46m'
    light_gray_background = '\033[47m'
    dark_gray_background = '\033[1000m'
    light_red_background = '\033[101m'
    light_green_background = '\033[102m'
    light_yellow_background = '\033[103m'
    light_blue_background = '\033[104m'
    light_magenta_background = '\033[105m'
    light_cyan_background = '\033[106m'
    white_background = '\033[107m'

    end = '\033[0m'


def color_print(string, color=Color.default, background=Color.default_background, **kwargs):
    """
    color and backgroud should be a member of the Color class
    :param string: the string to be printed
    :param color: text color
    :param background: text backgroud color
    :param kwargs: kwargs given to the normal print function
    """
    print(color + background + string + Color.end, **kwargs)


def bold_print(string, color=Color.default, background=Color.default_background, **kwargs):
    """
    color and backgroud should be a member of the Color class
    :param string: the string to be printed
    :param color: text color
    :param background: text backgroud color
    :param kwargs: kwargs given to the normal print function
    """
    print(Color.bold + color + background + string + Color.end, **kwargs)
