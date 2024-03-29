import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Read in all features data, including predictions
df = pd.read_csv('../data/output/predictions.csv')

# Add columns for pit, bench, blast
df['pit'] = df['hole_id'].str.slice(0,2)
df['bench'] = df['hole_id'].str.slice(3,6)
df['blast'] = df['hole_id'].str.slice(7,10)

# Separate dataset into train and predicted
train = df[df['data_type'] == 'train']
pre = df[df['data_type'] == 'predict']

label = 'litho_rock_class'

# Calculate appropriate position for annotation based on x, y
xmax = train['ActualX_mean'].max()
xmin = train['ActualX_mean'].min()
ymax = train['ActualY_mean'].max()
ymin = train['ActualY_mean'].min()
#x_annot = (xmax+xmin)/2
x_annot = xmax
y_annot = ymin

# Define colours for each rock class
colour_map = {'IF': 'rgb(200, 0, 0)',
             'AMP': 'rgb(175, 0, 255)',
             'QZ': 'rgb(252, 255, 0)',
             'LIM': 'rgb(125, 75, 32)',
             'GN': 'rgb(0, 255, 0)'}


def create_pred_scatters(pre, train):
    scatters = []
   
   # Add predicted holes to plot at full opacity
    for i in pre.pred.unique():
        splot = go.Scatter(
            x=pre[pre['pred'] == i]['ActualX_mean'],
            y=pre[pre['pred'] == i]['ActualY_mean'],
            text= ['PREDICTED BY MODEL<br>Prediction: {0} ({1:.0%} likelihood)'.format(pred, proba) + '<br>Exploration: {}'.format(exp) + '<br>Hole ID: {}'.format(hole_id)
                for pred, exp, proba, hole_id in zip(pre[pre['pred'] == i]['pred'], pre[pre['pred'] == i]['exp_rock_class'], pre[pre['pred'] == i][i], pre[pre['pred'] == i]['hole_id'])],
            hoverinfo='text',
            hoverlabel={'bgcolor':'#FBFCBD'},
            mode='markers',
            opacity=1,
            marker={
                'size': 10,
                'line': {'width': 0.5, 'color': 'white'},
                'color': colour_map[i]
            },name=i
        )
        scatters.append(splot)

    # Add training holes to plot for context in reduced opacity
    for  i in train[label].unique():
        splot = go.Scatter(
            x=train[train[label] == i]['ActualX_mean'],
            y=train[train[label] == i]['ActualY_mean'],
            text= ['USED FOR TRAINING<br>Litho Rock Class: {}'.format(rclass) + '<br>Hole ID: {}'.format(hole_id)
                for rclass, hole_id in zip(train[train[label] == i][label], train[train[label] == i]['hole_id'])],
            hoverinfo='text',
            hoverlabel={'bgcolor':'#FFFFFF'},
            mode='markers',
            opacity=0.3,
            marker={
                'size': 10,
                'line': {'width': 0.5, 'color': 'white'},
                'color': colour_map[i]
            },name=i
        )
        scatters.append(splot)

    return scatters           

app.layout = html.Div(style={'font-family': 'helvetica'}, children=[
    html.H1(
            children='ROCK CLASS PREDICTION DASHBOARD',
            style={
                    'textAlign': 'left',
                    'color': '#989898',
                    'font-family': 'helvetica',
                    'font-weight': 'bold'
                    }
    ),

    html.Br(),
    # Drop down to filter by pit, bench or blast
    html.Div([
    html.Div([
            # Radio button to toggle type of plot shown (predictions or likelihood)
            html.Span('Plot Shown:'),
            dcc.RadioItems(
                id='plot_shown',
                options=[
                    {'label': 'Predictions', 'value': 'predictions'},
                    {'label': 'Likelihood', 'value': 'likelihood'}
                ],
                value='predictions',
                labelStyle={'display': 'inline-block'}
            ),
            html.Br(),

            # Dropdown selections for prediction plot
            html.Span('PREDICTIONS'),
            html.Br(),
            html.Div([html.Span('Select a pit:'),
            dcc.Dropdown(
                id='pit_select',
                placeholder='All pits',
                multi=False,
                options=[{'label': i, 'value': i} for i in pre['pit'].unique()])]),

            html.Br(),
            html.Div([html.Span('Select a bench:'),
            dcc.Dropdown(
                id='bench_select',
                placeholder='All benches',
                multi=False,
                options=[{'label': i, 'value': i} for i in pre['bench'].unique()])]),

            html.Br(),
            html.Div([html.Span('Select a blast:'),
            dcc.Dropdown(
                id='blast_select',
                placeholder='All blasts',
                multi=False,
                options=[{'label': i, 'value': i} for i in pre['blast'].unique()])]),

            # Dropdown selections for likelihood plot
            html.Br(),
            html.Span('LIKELIHOOD'),
            html.Br(),
            # Dropdown to choose which rock class to visualize probabilities
            html.Div([html.Span('Select a rock class:'),
                dcc.Dropdown(
                    id='proba_select',
                    value=pre.pred.unique()[0],
                    multi=False,
                    options=[{'label': i, 'value': i} for i in pre.pred.unique()]
                    )]),
            ], style={'width': '10%', 'height':'100%', 'float': 'left', 'padding-left':10}),

    # Initial plot shown is scatterplot of rock class predictions by X, Y location
    dcc.Graph(
        id='main_plot',
        style={'width': '87%', 'height': '80vh', 'float':'right'},
        figure={
            'data': create_pred_scatters(pre, train),
            'layout': go.Layout(
                xaxis={'title': 'X'},
                yaxis={'title': 'Y'},
                #margin={'l': 80, 'b': 40, 't': 10, 'r': 50},
                legend={'x': 1, 'y': 1},
                hovermode='closest',
                annotations=[dict(
                        x=x_annot,
                        y=y_annot,
                        text='Predictions are shown in full opacity and training data in lighter colour',
                        xref='x',
                        yref='y',
                        showarrow=False
                    )]
            )
        }
    )
    ], style={'width': '100%', 'display': 'inline-block', 'margin':'auto', 'height': '100%'})
])

@app.callback(
   dash.dependencies.Output('main_plot', 'figure'),
   [dash.dependencies.Input('pit_select', 'value'),
   dash.dependencies.Input('bench_select', 'value'),
   dash.dependencies.Input('blast_select', 'value'),
   dash.dependencies.Input('plot_shown', 'value'),
   dash.dependencies.Input('proba_select', 'value')])
def update_plot(pit, bench, blast, plot_type, rock_class):
    tdf = train.copy()
    pdf = pre.copy()

    # Filter data based on selections
    if pit is not None:
        tdf = tdf[tdf['pit'] == pit]
        pdf = pdf[pdf['pit'] == pit]
    if bench is not None:
        tdf = tdf[tdf['bench'] == bench]
        pdf = pdf[pdf['bench'] == bench]
    if blast is not None:
        tdf = tdf[tdf['blast'] == blast]
        pdf = pdf[pdf['blast'] == blast]

    # Produces scatterplot of probabilities for 1 rock class
    if plot_type == "likelihood":
        return {
            'data': [
                go.Scatter(
                    x=pdf['ActualX_mean'],
                    y=pdf['ActualY_mean'],
                    text= ['{0:.0%} likelihood'.format(proba) + '<br>Hole ID: {}'.format(hole_id)
                            for proba, hole_id in zip(pdf[rock_class], pdf['hole_id'])],
                    hoverinfo='text',
                    hoverlabel={'bgcolor':'#FBFCBD'},
                    mode='markers',
                    #opacity=0.7,
                    marker={
                        'size': 10,
                        'line': {'width': 0.5, 'color': 'white'},
                        'color': pdf[rock_class],
                        'colorscale': [[0, 'rgba(255, 255, 255, 0.85)'], [1, 'rgba(20,25,137,1)']], # 0% white to 100% blue for  proba
                        "colorbar": {"title": 'Likelihood',
                                     "titleside": 'top',
                                     "tickmode": 'array',
                                     "tickvals": [0.01, 0.25, 0.5, 0.75, 0.99],
                                     "ticktext": ['0%', '25%', '50%', '75%', '100%'],
                                     "ticks": 'outside'}
                    },
                )
            ],
            'layout': go.Layout(
                title="Predicted Likelihood",
                titlefont=dict(
                    size=22
                ),
                xaxis={'title': 'X'},
                yaxis={'title': 'Y'},
                margin={'l': 80, 'b': 30, 't': 35, 'r': 50},
                legend={'x': 1, 'y': 1},
                hovermode='closest'
            )
        }

    # Updates scatterplot of rock class predictions
    if plot_type == "predictions":
        return {
                'data': create_pred_scatters(pdf, tdf),
                'layout': go.Layout(
                    title="Predicted Rock Class",
                    titlefont=dict(
                        size=22
                    ),
                    xaxis={'title': 'X'},
                    yaxis={'title': 'Y'},
                    margin={'l': 80, 'b': 30, 't': 35, 'r': 50},
                    legend={'x': 1, 'y': 1},
                    hovermode='closest', 
                    annotations=[dict(
                        x=x_annot,
                        y=y_annot,
                        text='Predictions are shown in full opacity and training data in lighter colour',
                        xref='x',
                        yref='y',
                        showarrow=False
                    )]
                )
            }


if __name__ == '__main__':
    app.run_server(debug=True)
