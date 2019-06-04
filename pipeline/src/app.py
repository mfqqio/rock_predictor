import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('../data/predictions.csv')

# Add columns for pit, bench, blast
df['pit'] = df['hole_id'].str.slice(0,2)
df['bench'] = df['hole_id'].str.slice(3,6)
df['blast'] = df['hole_id'].str.slice(7,10)

# Define colours for each rock class
colour_map = {'IF': 'rgb(200, 0, 0)', 
             'AMP': 'rgb(175, 0, 255)', 
             'QZ': 'rgb(252, 255, 0)', 
             'LIM': 'rgb(125, 75, 32)',
             'GN': 'rgb(0, 255, 0)'}
  
# Shows and updates scatterplot of probabilities for 1 rock class
def update_proba_plot(rock_class):   
    return dcc.Graph(
    id='proba_plot',
    figure={
        'data': [
            go.Scatter(
                x=df['ActualX_mean'],
                y=df['ActualY_mean'],
                text= ['{0:.0%} likelihood'.format(proba) + '<br>Hole ID: {}'.format(hole_id)
                        for proba, hole_id in zip(df[rock_class], df['hole_id'])],
                hoverinfo='text',
                hoverlabel={'bgcolor':'#FBFCBD'},
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 10,
                    'line': {'width': 0.5, 'color': 'white'},
                    'color': df[rock_class],
                    'colorscale': [[0, 'rgba(255, 255, 255, 0.85)'], [1, 'rgba(20,25,137,1)']], # white to blue
                    "colorbar": {"title": 'Likelihood',
                                 "titleside": 'top',
                                 "tickmode": 'array',
                                 "tickvals": [0, 0.25, 0.5, 0.75, 1],
                                 "ticktext": ['0%', '25%', '50%', '75%', '100%'],
                                 "ticks": 'outside'}
                },
            )
    
        ],
        'layout': go.Layout(
            xaxis={'title': 'X'},
            yaxis={'title': 'Y'},
            margin={'l': 80, 'b': 40, 't': 10, 'r': 50},
            legend={'x': 1, 'y': 1},
            hovermode='closest'
        )
    }
   )          
      
    
    
app.layout = html.Div(style={'font-family': 'helvetica'}, children=[
    html.H1(
            children='Rock Class Prediction Dashboard',
            style={
                    'textAlign': 'center'
                    }
    ),
    
    html.H3(
            children='Predicted Rock Classes',
            style={
                    'textAlign': 'center'
                    }
    ),
    
    # Drop down to filter by pit, bench or blast 
    html.Div([
            html.Div([html.Span('Select a pit:'),
            dcc.Dropdown(
                id='pit_select',
                placeholder='All pits',
                multi=False,
                options=[{'label': i, 'value': i} for i in df['pit'].unique()])], 
                    style={'width': '25%', 'float': 'left'}),
            
            html.Div([html.Span('Select a bench:'),
            dcc.Dropdown(
                id='bench_select',
                placeholder='All benches',
                multi=False,
                options=[{'label': i, 'value': i} for i in df['bench'].unique()])], 
                    style={'width': '25%', 'float': 'left'}),
    
            html.Div([html.Span('Select a blast:'),
            dcc.Dropdown(
                id='blast_select',
                placeholder='All blasts',
                multi=False,
                options=[{'label': i, 'value': i} for i in df['blast'].unique()])], 
                    style={'width': '25%', 'float': 'left'})
            ], style={'width': '100%', 'display': 'inline-block', 'marging':'auto', 'padding-left':150}),
    
    # Scatterplot of rock class predictions by X, Y location
    dcc.Graph(
        id='pred_plot',
        figure={
            'data': [
                go.Scatter(
                    x=df[df['pred'] == i]['ActualX_mean'],
                    y=df[df['pred'] == i]['ActualY_mean'],
                    text= ['Prediction: {0} ({1:.0%} likelihood)'.format(pred, proba) + '<br>Exploration: {}'.format(exp) + '<br>Hole ID: {}'.format(hole_id)
                        for pred, exp, proba, hole_id in zip(df[df['pred'] == i]['pred'], df[df['pred'] == i]['exp_rock_class'], df[df['pred'] == i][i], df[df['pred'] == i]['hole_id'])],
                    hoverinfo='text',
                    hoverlabel={'bgcolor':'#FBFCBD'},
                    mode='markers',
                    marker={
                        'size': 10,
                        'line': {'width': 0.5, 'color': 'white'},
                        'color': colour_map[i] 
                    },
                    name=i
                ) for i in df.pred.unique()
            ],
            'layout': go.Layout(
                xaxis={'title': 'X'},
                yaxis={'title': 'Y'},
                margin={'l': 80, 'b': 40, 't': 10, 'r': 50},
                legend={'x': 1, 'y': 1},
                hovermode='closest'
            )
        }
    ), 
   
    html.H3(
            children='Predicted Likelihood',
            style={
                    'textAlign': 'center'
                    }
    ),
    
    # Dropdown to choose which rock class to visualize probabilities  
    html.Div([html.Span('Select a rock class:'),
        dcc.Dropdown(
            id='proba_select',
            value=df.pred.unique()[0],
            multi=False,
            options=[{'label': i, 'value': i} for i in df.pred.unique()]
            
        )], style={'width': '30%', 'margin': 'auto'}),
        
    # Visualization of probabilities for 1 rock class
    html.Div([
        update_proba_plot(df.pred.unique()[0])])
])

@app.callback(
   dash.dependencies.Output('proba_plot', 'figure'),
   [dash.dependencies.Input('proba_select', 'value')])  
def update_proba_plot(rock_class):
    return {
        'data': [
            go.Scatter(
                x=df['ActualX_mean'],
                y=df['ActualY_mean'],
                text= ['{0:.0%} likelihood'.format(proba) + '<br>Hole ID: {}'.format(hole_id)
                        for proba, hole_id in zip(df[rock_class], df['hole_id'])],
                hoverinfo='text',
                hoverlabel={'bgcolor':'#FBFCBD'},
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 10,
                    'line': {'width': 0.5, 'color': 'white'},
                    'color': df[rock_class],
                    'colorscale': [[0, 'rgba(255, 255, 255, 0.85)'], [1, 'rgba(20,25,137,1)']], # white to blue
                    "colorbar": {"title": 'Likelihood',
                                 "titleside": 'top',
                                 "tickmode": 'array',
                                 "tickvals": [0, 0.25, 0.5, 0.75, 1],
                                 "ticktext": ['0%', '25%', '50%', '75%', '100%'],
                                 "ticks": 'outside'}
                },
            )
    
        ],
        'layout': go.Layout(
            xaxis={'title': 'X'},
            yaxis={'title': 'Y'},
            margin={'l': 80, 'b': 40, 't': 10, 'r': 50},
            legend={'x': 1, 'y': 1},
            hovermode='closest'
        )
    }
    
    
@app.callback(
   dash.dependencies.Output('pred_plot', 'figure'),
   [dash.dependencies.Input('pit_select', 'value'),
   dash.dependencies.Input('bench_select', 'value'),
   dash.dependencies.Input('blast_select', 'value')])  
def update_pred_plot(pit, bench, blast): # Update plot of rock predictions based on dropdown selectios
    dff = df.copy()
    if pit is not None:   
        dff = dff[dff['pit'] == pit]
    if bench is not None:
        dff = dff[dff['bench'] == bench]
    if blast is not None:
        dff = dff[dff['blast'] == blast]
    
    return {
            'data': [
                go.Scatter(
                    x=dff[dff['pred'] == i]['ActualX_mean'],
                    y=dff[dff['pred'] == i]['ActualY_mean'],
                    text= ['Prediction: {0} ({1:.0%} likelihood)'.format(pred, proba) + '<br>Exploration: {}'.format(exp) + '<br>Hole ID: {}'.format(hole_id)
                        for pred, exp, proba, hole_id in zip(dff[dff['pred'] == i]['pred'], dff[dff['pred'] == i]['exp_rock_class'], dff[dff['pred'] == i][i], dff[dff['pred'] == i]['hole_id'])],
                    hoverinfo='text',
                    hoverlabel={'bgcolor':'#FBFCBD'},
                    mode='markers',
                    marker={
                        'size': 10,
                        'line': {'width': 0.5, 'color': 'white'},
                        'color': colour_map[i] 
                    },
                    name=i
                ) for i in dff.pred.unique()
            ],
    
            'layout': go.Layout(
                xaxis={'title': 'X'},
                yaxis={'title': 'Y'},
                margin={'l': 80, 'b': 40, 't': 10, 'r': 50},
                legend={'x': 1, 'y': 1},
                hovermode='closest'
            )
        }
    
    
if __name__ == '__main__':
    app.run_server(debug=True)