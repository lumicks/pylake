from pygments.style import Style
from pygments.token import (
    Comment,
    Error,
    Generic,
    Name,
    Number,
    Operator,
    String,
    Text,
    Whitespace,
    Keyword,
)

colors = {
    "white": "#ffffff",
    "gray": "#404041",
    "mediumgray": "#929294",
    "lightgray": "#d9d9d9",
    "lightestgray": "#f0f0f0",
    "pink": "#d30359",
    "green": "#569a45",
    "blue": "#0049a9",
    "orange": "#e06700",
    "lightpink": "#f9d1d5",
    "lightgreen": "#cee6c8",
    "lightblue": "#a5ddf4",
    "lightorange": "#fcdfb7",
}


class PylakeStyle(Style):
    """
    Syntax highlighting with the pylake logo colors
    """

    background_color = colors["lightestgray"]

    styles = {
        Comment: f'italic {colors["mediumgray"]}',
        Comment.Preproc: "noitalic",
        Comment.Special: "bold",
        Error: f'bg:{colors["pink"]} {colors["white"]}',
        Generic.Deleted: f'border:{colors["pink"]} bg:{colors["lightpink"]}',
        Generic.Emph: "italic",
        Generic.Error: colors["pink"],
        Generic.Heading: f'bold {colors["blue"]}',
        Generic.Inserted: "border:{} bg:{}".format(colors["green"], colors["lightgreen"]),
        Generic.Output: colors["gray"],
        Generic.Prompt: f'bold {colors["blue"]}',
        Generic.Strong: "bold",
        Generic.Subheading: f'bold {colors["blue"]}',
        Generic.Traceback: colors["pink"],
        Keyword: f'bold {colors["green"]}',
        Keyword.Pseudo: "nobold",
        Keyword.Type: colors["pink"],
        Name.Attribute: f'italic {colors["blue"]}',
        Name.Builtin: f'bold {colors["green"]}',
        Name.Class: "underline",
        Name.Namespace: f'bold {colors["blue"]}',
        Name.Constant: colors["orange"],
        Name.Decorator: f'bold {colors["orange"]}',
        Name.Entity: f'bold {colors["pink"]}',
        Name.Exception: f'bold {colors["pink"]}',
        Name.Function: f'bold {colors["blue"]}',
        Name.Tag: f'bold {colors["blue"]}',
        Number: f'{colors["orange"]}',
        Operator: colors["blue"],
        Operator.Word: f'bold {colors["blue"]}',
        String: f'{colors["pink"]}',
        String.Doc: "italic",
        String.Escape: f'bold {colors["pink"]}',
        String.Other: colors["orange"],
        String.Symbol: f'bold {colors["pink"]}',
        Text: colors["gray"],
        Whitespace: colors["lightestgray"],
    }
