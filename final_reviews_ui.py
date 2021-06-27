# IMPORTING LIBRARIES
import pickle
import pandas as pd
import webbrowser

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output , State

import plotly.express as px

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer  

import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import base64

# READING AND PREPARING DATA
df = pd.read_csv('scrappedreviews.csv')
df.head()
df.shape

reviews = list(df.Reviews)

# LOADING PICKLE MODELS
file = open("pickle_model1.pkl", 'rb')
pickle_model = pickle.load(file, encoding = ('ISO-8859-1'))

file1 = open("features1.pkl", 'rb')
vocab = pickle.load(file1, encoding = ('ISO-8859-1'))

# Function checking whether a given review is Positive or Negative
def check_review(reviewText):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace", vocabulary = vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
   
    return pickle_model.predict(vectorised_review)

# Function for directly opening the browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

# Preparing data for Pie-Chart and Wordcloud
output_list = list(map(check_review, reviews))

# Pie-Chart
ratings = output_list
positive = []
negative = []

for rating in ratings:
    if (rating == 0):
        negative.append(rating)
    else:
        positive.append(rating)

perc_p = int(((len(positive)/len(ratings))*100))
perc_n = int(((len(negative)/len(ratings))*100))

data = {'Reviews': ['Positive', 'Negative'], 'Percentage': [perc_p, perc_n]}
df1 = pd.DataFrame(data)

sreviews = str(reviews)

df2 = pd.read_csv('scrappedreviews.csv')
df2 = df2['Reviews']

# Wordcloud
wc = WordCloud(max_font_size = 80, max_words = 100, width = 800, height = 400, background_color="black").generate(sreviews)
wc.to_file('img.png')

image_filename = 'img.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# MAIN LAYOUT OF APPLICATION
app = dash.Dash(__name__, external_stylesheets = [dbc.themes.CYBORG])

app.layout = html.Div(children = [
    html.Div(children = [
    html.H1(id = 'Main_title', children = 'SENTIMENT ANALYSIS WITH INSIGHTS', style={
    'textAlign': 'center',
    'color': 'white',
    'font-size': 50,
    'margin-top': '20px'}),
    html.H1(children = 'DASHBOARD', style = {'backgroundColor': 'yellow', 'textAlign': 'center', 'color': 'black'})
    ]),
    
    html.H1(children = "Select any review from the dropdown list and check whether it's Positive or Negative", style = {'font-size': 20, 'margin-top': '80px', 'font-style': 'bold', 'color': 'white'}),
    html.Div([dcc.Dropdown(id = 'dropdown',
                           options = [{'label': v[:260], 'value': v} for v in df.Reviews],
                           style = {'width':'98%', 'height':50, 'margin': '10px'},
                           searchable = False),
                           html.H1(children = None, id='result1', style = {'margin': '10px', 'margin-bottom': '40px'})]),
    
    html.Div(children = [html.Div(className = 'row',               
    children = [html.Div(className = 'six columns div-user-controls bg-grey',
             children = [html.H1(children = '''PIE-CHART''', style = {'color': 'yellow'}),
                         html.H1(children = "Showing Positive and Negative Reviews as percentages", style = {'font-size': 15, 'font-style': 'italic', 'color': 'white'}),
                         html.Div([dcc.Graph(id = 'pie-chart',
                                             figure = px.pie(df1, values='Percentage', names = 'Reviews', title='POSITIVE AND NEGATIVE REVIEWS'),
                                             style={'height': 400, 'width':600, 'margin-top': '25px', 'margin-bottom': '80px', 'margin': '10px'})])]),
                         html.Div(className = 'six columns div-user-controls bg-grey',
                                  children = [html.H1(children = 'WORDCLOUD', style = {'color': 'yellow'}),
                                              html.H1(children = "Showing the most frequently used words in the list of Reviews", style = {'font-size': 15, 'font-style': 'italic', 'color': 'white'}),
                                              html.Img(id = 'wordcloud',
                                                       src = app.get_asset_url('img.png'), 
                                                       style={'height': 400, 'width':600, 'margin-top': '25px', 'margin-bottom': '80px', 'margin': '10px'})])])]),
                      
        
    html.H1(children = "Enter any review of your choice and check whether it's Positive or Negative", style = {'font-size' : 20, 'margin-top': '80px', 'font-style': 'bold', 'color': 'white'}),      
    html.H1([dcc.Textarea(id = 'textarea_review',
                           placeholder = 'Enter the review here.....',
                           style = {'width':'98%', 'height':100, 'margin': '10px'}),
   
              dbc.Button(children = 'Find Review',
                         id = 'button_review',
                         color = 'success',
                         style= {'width':'98%', 'margin': '10px', 'color': 'black'}),
                         html.H1(children = None, id='result')])], style = {'backgroundColor': 'lightblack'})
                      

# FUNCTION CALLBACKS
# Callback 1
@app.callback(
    Output('result', 'children'),
    [Input('button_review', 'n_clicks')],
    [State('textarea_review', 'value')])
def update_output(n_clicks, textarea_value):

    if (n_clicks > 0):

        response = check_review(textarea_value)
        if (response[0] == 0):
            result = 'NEGATIVE'
        elif (response[0] == 1 ):
            result = 'POSITIVE'
        else:
            result = 'UNKNOWN'
       
        return result
       
    else:
        return ""

# Callback 2
@app.callback(
    dash.dependencies.Output('result1', 'children'),
    [dash.dependencies.Input('dropdown', 'value')])
def update_output1(value):
   
    response1 = check_review(value)
    if (response1[0] == 0):
        result1 = 'NEGATIVE'
    elif (response1[0] == 1 ):
        result1 = 'POSITIVE'
    else:
        result1 = 'UNKNOWN'
       
    return result1

# MAIN FUNCTION
if __name__ == '__main__':
    open_browser()
    app.run_server()









