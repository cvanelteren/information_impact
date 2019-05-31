#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:01:06 2019

@author: casper
"""

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

# default values
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.config.assets_folder = 'assets'     # The path to the assets folder.
app.config.include_asset_files = True   # Include the files in the asset folder
app.config.assets_external_path = ""    # The external prefix if serve_locally == False
app.config.assets_url_path = '/assets'  # the local url prefix ie `/assets/*.js`

app.layout = html.Div(
    [ html.H1("This is a test")]
)

if __name__ == '__main__':
    app.run_server(debug=True)