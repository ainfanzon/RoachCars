import os
import PIL
import ast
import base64
import signal
import random
import string
import psycopg2
import seaborn as sns
import numpy as np
import warnings
#import datetime
import subprocess
import shared_state as s
import ipywidgets as widgets
import matplotlib.pyplot as plt
#import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=FutureWarning)

from sql import sql
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
from io import BytesIO
from wordcloud import WordCloud
from datetime import datetime
from pkgs_import import *
from PIL import Image, ImageGrab
from imgbeddings import imgbeddings
from sentence_transformers import SentenceTransformer
from IPython.display import clear_output, IFrame, display, HTML, Markdown
from ipywidgets import GridspecLayout, Button, Layout, jslink, IntText, IntSlider, interact, HBox, VBox, Output

model = SentenceTransformer('all-MiniLM-L6-v2')

output_01 = widgets.Output()
output_02 = widgets.Output()
output_03 = widgets.Output()
output_04 = widgets.Output()

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def display_result(result):
    for row in result:
        fullpath = row[1]
        try:
            # Load and encode image
            img = Image.open(fullpath)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()

            # Create HTML layout
            html = f"""
                <div style="display: flex; align-items: center; gap: 20px;">
                    <img src="data:image/png;base64,{img_b64}" width="200" style="border:1px solid #ccc;" />
                    <div style="line-height: 1.6;">
                    <div><strong>VIN:</strong> {row[0]}
                </div>
                </div>
                    <div style="line-height: 1.6;">
                    <div><strong>Make:</strong> {row[2]}</div>
                    <div><strong>Model:</strong> {row[3]}</div>
                    <div><strong>Year:</strong> {row[4]}</div>
                    <div><strong>Price:</strong> ${int(row[5]):,}</div>
                    <div><strong>Distance between vectors:</strong> {float(row[6]):.5f}</div>
                    <div><strong>Closeness:</strong> {row[7]}%</div>
                </div>
            </div>
            <hr/>
            """
            display(HTML(html))
        except Exception as e:
            print(f"Error displaying image {fullpath}: {e}")
        
def generate_fake_vin():
    allowed_chars = string.ascii_uppercase.replace('I', '').replace('O', '').replace('Q', '') + string.digits
    return ''.join(random.choices(allowed_chars, k=17))

def get_random_L100(make, model):
    match make:
        case "Audi":
            result = f"5.0-10.0 L/100"
        case "Hyundai":
            result = f"6.0-12.0 L/100"
        case "Mahindra":
            result = f"5.0-12.5 L/100"
        case "RollsRoyce":
            result = f"15.0-19.0 L/100"
        case "Suzuki":
            result = f"4.5-8.0 L/100"
        case "Tata":
            result = f"6-14.0 L/100"
        case "Toyota":
            result = f"4-10.5 L/100"
        case "Tesla":
            result = f"6-14.0 L/100"
        case "Volkswagen":
            result = f"4-10.5 L/100"
        case _:
            result = 0
    return result

def generate_random_year(start_year, end_year):
    if start_year > end_year:
        raise ValueError("Start year must be less than or equal to the end year.")
    else:
        return random.randint(start_year, end_year)

def get_random_month():
    return str(random.randint(1, 12))

def get_random_registration(y):
    current_year = datetime.now().year
    if y == current_year:
        reg  = str(datetime.now().month) + '/' + str(current_year)
    else:
        reg  = str(get_random_month()) + '/' + str(generate_random_year(2000, 2024))

    return str(reg)

def get_random_fuel():
    return random.choice(fuel_type)

def get_random_pwrkw():
    return random.randint(150, 199)

def get_random_pwrps():
    return random.randint(190, 290)

def get_random_gkm():
    return random.randint(125, 250)

def get_random_milage(y):
    if y <= 2000:
        result = random.randint(75000, 150000)
    elif 2000 < y <= 2015:
        result = random.randint(25000, 75000)
    elif y > 2015:
        result = random.randint(1000, 25000)
    else:
        result = 0  # Optional fallback
    return result

def embed_text(text):
    embedding = model.encode(text)
    return embedding.tolist()  # convert numpy array to list for SQL insert
    
def get_random_transmission():
    return random.choice(transmissions)

def get_random_price(min_price=10000, max_price=50000):
    return random.randint(min_price, max_price)

def on_bt_from_clipboard_clicked(b):
    output_01.clear_output()
    output_02.clear_output()
    with output_01:
        s.img = ImageGrab.grabclipboard()                             # Capture the image from the clipboard
        s.vin = generate_fake_vin()                                   # Generate the VIN
        if s.img is None:
            print("No image found in the clipboard.")
        else:
            ibed = imgbeddings()                                      # Initialize the imgbeddings model
            s.qry_embedding = ibed.to_embeddings(s.img)[0].tolist()   # Generate the embedding
            display(s.img)                                            # Display the image

# Display the dropdown widget
def on_bt_save_clicked(b):
    output_02.clear_output()
    with output_02:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s.filename = f"{s.images_home}{txt_make.value}/{txt_make.value}_{timestamp}.jpg"
        s.img.convert('RGB').save(s.filename, 'JPEG')
        try:
            s.img.convert('RGB').save(s.filename, 'JPEG')
            print(f"File saved successfully as {s.filename}")
            ibed = imgbeddings()                                          # Initialize the imgbeddings model
            s.qry_embedding = ibed.to_embeddings(s.img)[0].tolist()       # Generate the embedding
        except Exception as e:
            print(f"Error saving file: {e}")
            
 
# Function to display the selected value
def on_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        print(f"Selected option: {change['new']}")

# Form 01 Start ---------------

bt_from_clipboard = widgets.Button(
    description='Get Image From Clipboard'
  , disabled=False
  , layout=Layout(width='25%', height='35px')
  , button_style='success'                                                           # 'success', 'info', 'warning', 'danger' or ''
  , tooltip='From Clipboard'
  , icon='check'                                                                     # (FontAwesome names without the `fa-` prefix)
)

bt_from_clipboard.on_click(on_bt_from_clipboard_clicked)

# Form 01 End ---------------

# Form 02 Start ---------------

sldr_result = widgets.IntSlider(
      min=1
    , max=20
    , value=1
    , description='Number Cars In Result: '
    , layout = { 'width': '800px'}
    , style = {'description_width': 'initial'}
)

int_range = widgets.IntRangeSlider(
      value=[10, 500]
    , min=10
    , max=500
    , step=1
    , description='USD Price Range (In Thousands):'
    , layout = { 'width': '750px'}
    , style = {'description_width': 'initial'}
    , continuous_update=False
)

# Form 02 End ---------------

# Form 03 Start ---------------

# Define the dropdown options

bt_save = widgets.Button(
    description='Save Image',
    disabled=False,
    # layout=Layout(width='25%', height='35px'),
    button_style='success',                                                         # 'success', 'info', 'warning', 'danger' or ''
    tooltip='From Clipboard',
    icon='check'                                                                    # (FontAwesome names without the `fa-` prefix)
)

bt_save.on_click(on_bt_save_clicked)

# Widgets for the form to capture attributes

# Create the dropdown widget
txt_make = widgets.Dropdown(
    options=s.dropdown_options 
  , description='Make:'
  , disabled=False
  , layout = { 'width': '200px'},
)

int_year = widgets.IntText(
    value=2025
  , description='Make Year:'
  , disabled=False
  , layout = { 'width': '150px'},
)

txt_model = widgets.Text(
    value=''
  , placeholder='Enter car model'
  , description='Model:'
  , disabled=False
  , layout = { 'width': '200px'},
)

txt_color = widgets.Text(
    value=''
  , placeholder='Color'
  , description='Color'
  , disabled=False   
  , layout = { 'width': '250px'},
)

int_price_usd = widgets.IntText(
    value = None
  , description='Price USD:'
  , disabled=False
  , layout = { 'width': '175px'},
)

int_milage = widgets.IntText(
    value = None
  , description='Milage:'
  , disabled=False
  , layout = { 'width': '175px'},
)

# Create the dropdown widget
dd_trans = widgets.Dropdown(
    options=s.transmission_values
  , description='Transmission:'
  , disabled=False 
  , layout = { 'width': '370px'}
  , style = {'description_width': 'initial'}
)

# Observe changes in the dropdown value
dd_trans.observe(on_change)

dd_fuel = widgets.Dropdown(
    options=s.fuel_types
  , description='Fuel Type:'
  , disabled=False
  , layout = { 'width': '190px'}
)


#lbl_reg = widgets.Label(
#      value = get_random_month() + "/" + str(int_year.value)
#)

#lbl_pwr_kw = widgets.Label(
#      value = str(get_random_pwrkw())
#)

#lbl_pwr_ps = widgets.Label(
#      value = str(get_random_pwrps())
#)

#lbl_L100 = widgets.Label(
#      value = get_random_L100(txt_make.value, txt_model.value)
#)

# Observe changes in the dropdown value
txt_make.observe(on_change)

s1r1 = HBox([bt_from_clipboard,])
s1r2 = HBox([output_01])
form_01 = VBox([s1r1, s1r2])

s2r1 = HBox([sldr_result])
s2r2 = HBox([int_range])
form_02 = VBox([s2r1, s2r2])

s3r1 = HBox([bt_from_clipboard,])
s3r2 = HBox([output_01])
s3r3 = HBox([txt_make, int_year, txt_model, txt_color, int_price_usd,int_milage,])
s3r4 = HBox([dd_trans, dd_fuel,])
s3r5 = HBox([output_02])
s3r6 = HBox([bt_save])
s3r7 = HBox([output_04])

form_03 = VBox([s3r1, s3r2, s3r7, s3r3, s3r4, s3r5, s3r6, s3r7,])