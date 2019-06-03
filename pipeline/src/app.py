import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('../data/predictions.csv')

# Subset and clean up data for display in app
display_df = df[['hole_id', 'exp_rock_type', 'exp_rock_class', 'pred', 'AMP', 'IF', 'LIM', 'QZ']]
display_df['AMP'] = display_df['AMP'] * 100
display_df['IF'] = display_df['IF'] * 100
display_df['LIM'] = display_df['LIM'] * 100
display_df['QZ'] = display_df['QZ'] * 100
display_df = display_df.round(2)
display_df.rename(columns={'hole_id': 'Hole ID', 
                           'exp_rock_type':'Exploration Model Rock Type',
                           'exp_rock_class': 'Exploration Model Rock Class',
                           'pred':'Predicted Rock Class',
                           'AMP':'AMP %',
                           'IF':'IF %',
                           'LIM':'LIM %',
                           'QZ':'QZ %'}, inplace=True)

# Shows raw data table
def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

# Shows and updates scatterplot of probabilities for 1 rock class
def update_proba_plot(rock_class):
    filtered_df = df[df.pred == rock_class]
    
    return dcc.Graph(
    id='proba_plot',
    figure={
        'data': [
            go.Scatter(
                x=df['ActualX_mean'],
                y=df['ActualY_mean'],
                text= ['{0:.0%} likelihood'.format(proba)
                        for proba in df[rock_class]],
                hoverinfo='text',
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 10,
                    'line': {'width': 0.5, 'color': 'white'},
                    #'color': 'rgba(102,0,102,1)',
                    'color': df[rock_class],
                    'colorscale': [[0, 'rgba(255, 255, 255, 0.85)'], [1, 'rgba(102,0,102,1)']], # white to purple
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
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 1, 'y': 1},
            hovermode='closest'
        )
    }
   )          
      
    
    
app.layout = html.Div([
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
    
    # Scatterplot of rock class predictions by X, Y location
    dcc.Graph(
        id='pred_plot',
        figure={
            'data': [
                go.Scatter(
                    x=df[df['pred'] == i]['ActualX_mean'],
                    y=df[df['pred'] == i]['ActualY_mean'],
                    text= ['Prediction: {}'.format(pred) + '<br>Exploration: {0} ({1:.0%} likelihood)'.format(exp, proba)
                        for pred, exp, proba in zip(df[df['pred'] == i]['pred'], df[df['pred'] == i]['exp_rock_class'], df[df['pred'] == i][i])],
                    hoverinfo='text',
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 10,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=i
                ) for i in df.pred.unique()
            ],
            'layout': go.Layout(
                xaxis={'title': 'X'},
                yaxis={'title': 'Y'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
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
    
    html.Div([
        # Dropdown to choose which rock class to visualize probabilities             
        html.Div([
            dcc.Markdown("Select Rock Class:"),
            dcc.Dropdown(id='proba_dropdown', options=[
            {'label': i, 'value': i} for i in df.pred.unique()
            ], multi=False, value=df.pred.unique()[0], style={'width': '25%', 'float':'left'}
        )]),   
        
        # Visualization of probabilities for 1 rock class
        html.Div([
            update_proba_plot('IF')], 
            style={'width': '75%', 'float':'right'})
    ], style={'width': '100%', 'display': 'inline-block'}),
          
   html.H4(children='Data by Individual Hole'),
   generate_table(display_df) 
])

if __name__ == '__main__':
    app.run_server(debug=True)