from typing import List, Dict, Any, Union, Literal

import ipywidgets as widgets
import plotly.graph_objs as go
import pandas as pd

from ipywidgets import Layout
from IPython.display import display, HTML

__all__ = ['create_display', 'create_tabs_widgets']

def display_table(data: pd.DataFrame, title: str = None, position: Literal["left", "center", "right"] = "left"):
    """Display data in ipython table format with title"""
    if title is not None:
        display(HTML(f'<h4 align="{position}"><u>{title}</u></h4>'))
    display(data)

def create_display(c: Union[pd.DataFrame, Dict[str, pd.DataFrame], go.Figure]):
    """ different kinds of display in jupyter notebook """
    if isinstance(c, pd.DataFrame):
        return display_table(c)
    if isinstance(c, go.Figure):
        return display(go.FigureWidget(c))
    return display(c)

def create_tabs_widgets(results: Dict[str, Union[Any, List[Any]]]) -> widgets.Tab:
    """ Create Tabs and display """
    output = {}
    for k, item in results.items():
        out = widgets.Output(layout=Layout(width="100%"))
        with out:
            if item is None:
                continue
            if isinstance(item, list):
                for i in item:
                    if i is not None:
                        create_display(i)
            else:
                create_display(item)
        output[k] = out

    tab = widgets.Tab(layout=Layout(width="100%"))
    tab.children = list(output.values())
    for i, k in enumerate(output.keys()):
        tab.set_title(i, k)

    return tab
