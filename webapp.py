import streamlit as st
import dill
import pandas as pd
import numpy as np
from unidecode import unidecode
from string import punctuation
import PIL
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchdistill.core.forward_hook import ForwardHookManager
import PIL
from torchvision.transforms.functional import pil_to_tensor
from haversine import haversine
import plotly.express as px
import pydeck as pdk
import matplotlib.pyplot as plt

############################################################### USEFUL FUNCTIONS
# function to extract feature maps from an apartment photo
def get_feature_maps_from_photo(file):
    
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    forward_hook_manager = ForwardHookManager(torch.device("cpu"))
    forward_hook_manager.add_hook(model, 'fc', requires_input=True, requires_output=False)

    resnet18_transforms = ResNet18_Weights.IMAGENET1K_V1.transforms(antialias=True)

    x = pil_to_tensor(PIL.Image.open(file).convert("RGB"))
    x = resnet18_transforms(x).unsqueeze(0).to(torch.device("cpu"))
    
    y = model.forward(x)
    
    io_dict = forward_hook_manager.pop_io_dict()
    
    feature_maps = io_dict["fc"]["input"].detach().numpy()

    return feature_maps

# function to preprocess the complete_description column
def preprocess_complete_description(x):
    x = unidecode(" ".join(x.lower().split()))
    for p in punctuation:
        x = x.replace(p, " ")
    for n in "0123456789":
        x = x.replace(n, " ")
    return " " + " ".join(x.split()) + " "

############################################################### LOADING USEFUL STUFF
    
# loading reference data
data_train_test = dill.load(open("5_modeling/final_model/data_train_test.pk", "rb"))
data_train = data_train_test["train"]
data_test = data_train_test["test"]
total_data = pd.concat((data_train, data_test))

# lat long cluster centers
centroids = data_train.groupby("lat_long_cluster")[["lat", "long"]].mean()

# loading model and predicting over test data for further analysis
model = dill.load(open("5_modeling/final_model/fitted_model.pk", "rb"))
data_test["price_predicted"] = 10**model.predict(data_test.drop(columns=["price"]))
data_test["log_price_predicted"] = model.predict(data_test.drop(columns=["price"]))
data_test["log_price"] = np.log10(data_test["price"])

# calculating errors
data_test["error"] = data_test["price_predicted"] - data_test["price"]
data_test["error_log"] = data_test["log_price_predicted"] - data_test["log_price"]
data_test["percentage_error"] = (data_test["price_predicted"] - data_test["price"])/data_test["price"]*100
data_test["log_percentage_error"] = (data_test["log_price_predicted"] - data_test["log_price"])/data_test["log_price"]*100


# feature names
cat_cols = [col for col in data_train.columns if "Cat_" in col]

# shopping centers
shoppings = {"Passeio_das_Aguas": (-16.62935338717902, -49.27864911348494),
            "Flamboyant": (-16.709983713886288, -49.237018433761605),
            "Portal_Sul": (-16.770457543653993, -49.35181898957849),
            "Goiânia": (-16.70811433888118, -49.27281973376176),
            "Cerrado": (-16.665992404732286, -49.308465857660444),
            "Portal": (-16.654047808420035, -49.32928694540062),
            "Plaza_Doro": (-16.70722528961973, -49.32804553190786),
            "Cidade_Jardim": (-16.682721816436608, -49.31411186259885),
            "Perimetral_Open_Mall": (-16.642368044349382, -49.307910909152895),
            "Araguaia": (-16.658321034298336, -49.25941657000837),
            "Bouganville": (-16.694169294923878, -49.265424718418366),
            "Buriti": (-16.74178147681141, -49.27701120492441),
            "Lozandes": (-16.692997718702827, -49.22246682530371),
            "Orion_Complex": (-16.69664826311222, -49.270025781651654),
            "Republica": (-16.67823083437392, -49.26693911841769),
            "Anhanguera": (-16.673622640794598, -49.2555401644532),
            "Gallo": (-16.661639216531704, -49.25507395766126),
            "Estacao_Goiania": (-16.66110701599241, -49.261788203072975),
            "Buena_Vista": (-16.710339841635147, -49.26776864725288)}

############################################################### SECTION STATE INFO

if 'disable_iptu' not in st.session_state:
    st.session_state.disable_iptu = False
if 'disable_condo_fee' not in st.session_state:
    st.session_state.disable_condo_fee = False
if 'log_x_histogram' not in st.session_state:
    st.session_state.log_x_histogram = False
if 'log_y_histogram' not in st.session_state:
    st.session_state.log_y_histogram = False

# function to update the disabled parameter from the sliders
def update_iptu_slider():
    st.session_state.disable_iptu = st.session_state.selectbox_iptu != 'No'
# function to update the disabled parameter from the sliders
def update_condo_fee_slider():
    st.session_state.disable_condo_fee = st.session_state.selectbox_condo_fee != 'No'
def update_log_x():
    st.session_state.log_x_histogram = st.session_state.selectbox_log_x_hist == 'No'
def update_log_y():
    st.session_state.log_y_histogram = st.session_state.selectbox_log_y_hist == 'No'

############################################################### THE WEBAPP STARTS HERE
# sidebar section selection
section = st.sidebar.selectbox("Section:", ["Apartment price prediction", "Data exploration", "Model performance analysis"])

# sidebar text
st.sidebar.markdown("""               
## 🎯 Objective of this project

To deploy a web application capable of providing predictions for apartment prices in **Goiânia**. All you need to do is **fill in the input data form and click "Predict!"**!

## 📄 Additional information

The static prediction model was trained using data collected from the internet (August, 2024), for **educational purposes only**. You should **NOT** use the model's predictions to make actual decisions or for commercial purposes. If you would like to better understand the steps that were taken to collect, clean and model the data, please consider visiting my [github repository](https://github.com/cego669/ApartmentPricesInGoiania).

## 📨 Contact me

[LinkedIn](https://www.linkedin.com/in/cego669/)

## ⚖️ License

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/cego669/ApartmentPricesInGoiania">ApartmentPricesInGoiania</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://www.linkedin.com/in/cego669/">Carlos Eduardo Gonçalves de Oliveira</a> is licensed under
<a href="https://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CCA-NC 4.0 International</a>.</p>
<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt="">
""", unsafe_allow_html=True)

###################################### SECTION: "Apartment price prediction"
if section == "Apartment price prediction":

    st.markdown("""
## 🔮 Apartment price prediction
                
Please fill out the form below to obtain a price prediction for the apartment.
Note that the model used for prediction is static and was trained with data collected from the internet in August 2024.
**The predictions are only applicable for apartments located in Goiânia, Goiás, Brazil**.

- You can explore the data used for training/testing the model in the "Data exploration" section.
- The model's performance on data outside the training set can be analyzed in the "Model performance analysis" section.
    """)

    cols = st.columns(2)
    st.session_state.selectbox_condo_fee = cols[0].selectbox("Do you want to inform the value of condominium fee?", ["Yes", "No"], on_change=update_condo_fee_slider)
    st.session_state.selectbox_iptu = cols[1].selectbox("Do you want to inform the value of the IPTU?", ["Yes", "No"], on_change=update_iptu_slider)
    with st.form(key = "user_input"):
        st.markdown("""
        <h3 style='text-align: center;'>Input data</h3>
        """, unsafe_allow_html=True)
        
        cols = st.columns(3)
        condo_fee = cols[0].number_input("Condominium fee (R$)", data_train["condo_fee"].min(),
                        data_train["condo_fee"].max(),
                        data_train["condo_fee"].median(),
                        step=1.0,
                        disabled=st.session_state.disable_condo_fee)
        iptu = cols[1].number_input("IPTU/year (R$)", data_train["iptu"].min(),
                        data_train["iptu"].max(),
                        data_train["iptu"].median(),
                        step=1.0,
                        disabled=st.session_state.disable_iptu)
        
        if st.session_state.disable_condo_fee:
            condo_fee = np.nan
        if st.session_state.disable_iptu:
            iptu = np.nan

        floorSize = cols[2].number_input("Floor size (m²)", data_train["floorSize"].min(),
                            data_train["floorSize"].max(),
                            data_train["floorSize"].median(),
                            step=1.0,
                            format="%f")
        
        cols = st.columns(2)
        numberOfRooms = cols[0].selectbox("Number of rooms", np.sort(np.unique(np.round(data_train["numberOfRooms"]))))
        numberOfBathroomsTotal = cols[1].selectbox("Number of bathrooms", np.sort(np.unique(np.round(data_train["numberOfBathroomsTotal"]))))

        cols = st.columns(2)
        numberOfParkingSpaces = cols[0].selectbox("Number of parking spaces", np.sort(np.unique(np.round(data_train["numberOfParkingSpaces"]))))
        floorLevel = cols[1].selectbox("Floor level", np.sort(np.unique(data_train["floorLevel"])))
        
        cat_list = [s.replace("Cat_", "") for s in cat_cols]
        cat_list.sort()
        Cat_ = st.multiselect("Specific characteristics", cat_list)
        
        cols = st.columns(2)
        st.markdown("You can discover the **latitude and longitude** from an address by using [Google Maps](https://www.google.com/maps)")
        lat = cols[0].number_input("Latitude (°)", data_train["lat"].min(),
                            data_train["lat"].max(),
                            data_train["lat"].mean(),
                            step=10**-9,
                            format="%.9f")
        long = cols[1].number_input("Longitude (°)", data_train["long"].min(),
                            data_train["long"].max(),
                            data_train["long"].mean(),
                            step=10**-9,
                            format="%.9f")
        refresh_map = st.columns(3)[1].form_submit_button("Refresh position in map")
        st.map(data=pd.DataFrame({"lat": [lat], "long":[long]}), latitude="lat", longitude="long", color="#16e8d9", use_container_width=True)

        file = st.file_uploader("Please upload the main photo used in the apartment listing ('.png', '.jpg', '.webp'):", type=["png", "jpg", "webp"])
        
        complete_description = st.text_input("Please paste the full description text used in the apartment listing (in portuguese):", value="", max_chars=None)

        predict = st.columns(5)[-1].form_submit_button("Predict!")

    if predict:
        # initializing user's input data
        X_user = {}

        # numerical input
        X_user["numberOfRooms"] = [numberOfRooms]
        X_user["numberOfBathroomsTotal"] = [numberOfBathroomsTotal]
        X_user["numberOfParkingSpaces"] = [numberOfParkingSpaces]
        X_user["iptu"] = [iptu]
        X_user["condo_fee"] = [condo_fee]
        X_user["floorSize"] = floorSize
        X_user["floorLevel"] = [floorLevel]

        # adding binary columns from the characteristics of the apartment
        for cat_col in cat_cols:
            if cat_col.replace("Cat_", "") in Cat_:
                X_user[cat_col] = [1]
            else:
                X_user[cat_col] = [0]

        # total specific characteristics for the apartment
        X_user["total_cat"] = [len(Cat_)]

        # getting lat_long_cluster
        lat_long_cluster = centroids.apply(lambda row: (row.lat - lat)**2 + (row.long - long)**2, axis=1)
        lat_long_cluster = centroids.index[lat_long_cluster == lat_long_cluster.min()][0]
        X_user["lat_long_cluster"] = [lat_long_cluster]

        # calculating distances to well known shopping centers in Goiânia
        for key in shoppings.keys():
            X_user[f"dist_{key}"] = [haversine((lat, long), shoppings[key])]

        # getting feature maps
        feature_maps = get_feature_maps_from_photo(file)
        
        for i in range(512):
            X_user[f"feature_maps_{i+1}"] = [feature_maps[0, i]]

        # preprocessing complete_description
        complete_description = preprocess_complete_description(complete_description)
        X_user["complete_description"] = [complete_description]

        # calculating description length
        description_length = len(complete_description)
        X_user["description_length"] = [description_length]

        # converting user's input data to a DataFrame
        X_user = pd.DataFrame(X_user)
        
        # getting prediction from model
        y_pred = np.round(10**model.predict(X_user)[0], 2)

        # printing y_pred
        st.success(f"Predicted price is R${y_pred}", icon="🔮")

###################################### SECTION: "Apartment price prediction"
elif section == "Data exploration":
    
    st.markdown("""
## 🕵🏽 Data exploration
                
In this section you can explore the data used to train/test the prediction model using interactive plots.
Remember that the data only includes **apartments located in Goiânia**.
Below, only numerical variables were available for exploration.
For a more detailed exploration, please consider visiting my [github repository](https://github.com/cego669/ApartmentPricesInGoiania).
    """)

    # select vis type
    vis_type = st.selectbox("Visualization type:", ["Histogram with boxplot", "Scatter plot with LOESS trendline", "Map"])

    # get key from value in a dictionary
    @st.cache_data
    def get_key(my_dict, val):
            for key, value in my_dict.items():
                if val == value:
                    return key

    # col formatting dict
    col_format_dict = {'floorLevel':"Floor level", 'condo_fee':"Condominium fee (R$)", 'iptu':"IPTU (R$)", 'floorSize':"Floor size (m²)",
                       'numberOfRooms':"Number of rooms", 'numberOfBathroomsTotal':"Number of bathrooms", 'numberOfParkingSpaces':"Number of parking spaces",
                       "price":"Price (R$)"}
    
    # get color scale from values of a column
    @st.cache_data
    def get_color_scale(values):
        max_value = values.max()
        min_value = values.min()
        return [((values.quantile(.25)-min_value)/(max_value-min_value), "blue"),
                ((values.quantile(.75)-min_value)/(max_value-min_value), "red")]
    
    # function to normalize values
    @st.cache_data
    def normalize_values(values):
        return (values - np.quantile(values, .05)) / (np.quantile(values, .95) - np.quantile(values, .05))

    # function to convert a normalized value to rgb
    @st.cache_data
    def value_to_rgb(value_normalized):
        # blue -> [0, 0, 255], red -> [255, 0, 0]
        r = int(255 * value_normalized)/255     # increases red value
        g = 100/255                               # fixed green
        b = int(255 * (1 - value_normalized))/255  # decreases blue value
        alpha = .5
        return (r, g, b, alpha)
    
    # histogram with boxplot
    if vis_type == "Histogram with boxplot":

        col = st.selectbox("Column for X:", col_format_dict.values())
        st.session_state.selectbox_log_x_hist = st.selectbox("Do you want to do log(x)?", ["No", "Yes"], on_change=update_log_x)

        fig = px.histogram(x=total_data[get_key(col_format_dict, col)], log_x=st.session_state.log_x_histogram, marginal="box")
        fig.update_layout(xaxis_title=col, yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    #  scatter plot with a LOESS trendline
    if vis_type == "Scatter plot with LOESS trendline":
        
        st_cols = st.columns(2)
        st.session_state.selectbox_log_x_hist = st_cols[0].selectbox("Do you want to do log(x)?", ["No", "Yes"], on_change=update_log_x)
        st.session_state.selectbox_log_y_hist = st_cols[1].selectbox("Do you want to do log(y)?", ["No", "Yes"], on_change=update_log_y)

        # selection of data for visualization
        st_cols = st.columns(3)
        col1 = st_cols[0].selectbox("Column for X:", col_format_dict.values())
        col2  = st_cols[1].selectbox("Column for Y:", col_format_dict.values(), len(col_format_dict.values())-1)
        col3 = st_cols[2].selectbox("Column for color:", [None] + list(col_format_dict.values()))

        # if user dont want to see the color...
        if col3 == None:
            fig = px.scatter(x=total_data[get_key(col_format_dict, col1)], y=total_data[get_key(col_format_dict, col2)],
                         log_x=st.session_state.log_x_histogram, log_y=st.session_state.log_y_histogram,
                         trendline="lowess")
        # if user want to see the color...
        else:
            fig = px.scatter(x=total_data[get_key(col_format_dict, col1)], y=total_data[get_key(col_format_dict, col2)],
                            log_x=st.session_state.log_x_histogram, log_y=st.session_state.log_y_histogram,
                            trendline="lowess", color=total_data[get_key(col_format_dict, col3)],
                            color_continuous_scale=get_color_scale(total_data[get_key(col_format_dict, col3)]))
        
        # updating layout with useful features
        fig.update_layout(xaxis_title=col1, yaxis_title=col2)
        fig.data[1].update(line_color='#9429e3')
        st.plotly_chart(fig, use_container_width=True)
    
    if vis_type == "Map":
        
        col = st.selectbox("Column for color:", col_format_dict.values(), len(col_format_dict.values())-1)

        temp_data = total_data.dropna(subset=get_key(col_format_dict, col)).copy()
        normalized_values = normalize_values(total_data[get_key(col_format_dict, col)].dropna())
        rgb_colors = [value_to_rgb(val) for val in normalized_values]
        temp_data["color"] = rgb_colors
        
        st.map(temp_data, latitude="lat", longitude="long", color="color")

        # now we create a barcolor with plt
        fig, ax = plt.subplots(figsize=(6, 0.5))
        fig.subplots_adjust(bottom=0.5)

        # generating barcolor based on the values from col
        cmap = plt.cm.bwr  # choosing color gradient
        norm = plt.Normalize(vmin=temp_data[get_key(col_format_dict, col)].quantile(.05), vmax=temp_data[get_key(col_format_dict, col)].quantile(.95))

        # create colorbar
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal', label=col, format='%.2f')
        cbar.ax.tick_params(labelrotation=45)
        # exhibits the colorbar
        st.pyplot(fig)

###################################### SECTION: "Apartment price prediction"
if section == "Model performance analysis":

    st.markdown("""
## 📊 Model performance analysis
                
Here you can explore how the prediction model performed on test data using interactive plots and additional information.
""")
    
    rmse = np.sqrt(np.mean((data_test["price"] - data_test["price_predicted"])**2))
    rmse2 = np.sqrt(np.mean((np.log10(data_test["price"]) - np.log10(data_test["price_predicted"]))**2))

    mae = np.mean(np.absolute(data_test["price"] - data_test["price_predicted"]))
    mae2 = np.mean(np.absolute(np.log10(data_test["price"]) - np.log10(data_test["price_predicted"])))

    mpe = np.mean(np.absolute(data_test["price"] - data_test["price_predicted"])/data_test["price"])
    mpe2 = np.mean(np.absolute(np.log10(data_test["price"]) - np.log10(data_test["price_predicted"]))/np.log10(data_test["price"]))

    st.markdown("""
#### Summary

The model was trained to predict apartment prices on a **logarithmic scale**.
Despite this, performance metrics on a natural scale will also be shown.

In natural scale:

- Root Mean Squared Error: R${:.2f}
- Mean Absolute Error: R${:.2f}
- Mean Percentage Error: {:.2f}%

In logarithmic scale:

- Root Mean Squared Error: {:.2f}
- Mean Absolute Error: {:.2f}
- Mean Percentage Error: {:.2f}%

#### Residual plots
""".format(rmse, mae, mpe*100,
           rmse2, mae2, mpe2*100))

    # select vis type
    vis_type = st.selectbox("Visualization type:", ["Histogram: error in natural scale",
                                                    "Histogram: error in log scale",
                                                    "Scatter plot: y_pred vs y_test (natural scale)",
                                                    "Scatter plot: y_pred vs y_test (log scale)",
                                                    "Scatter plot: percentage error (natural scale)",
                                                    "Scatter plot: percentage error (logarithmic scale)"])
    
    if vis_type == "Histogram: error in natural scale":
        fig = px.histogram(x=data_test["error"])
        fig.update_layout(xaxis_title="Error in natural scale", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    elif vis_type == "Histogram: error in log scale":
        fig = px.histogram(x=data_test["error_log"])
        fig.update_layout(xaxis_title="Error in logarithmic scale", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    elif vis_type == "Scatter plot: y_pred vs y_test (natural scale)":
        fig = px.scatter(x=data_test["price"], y=data_test["price_predicted"], trendline="lowess")
        fig.update_layout(xaxis_title="Price (R$)", yaxis_title="Predicted price (R$)")
        fig.data[1].update(line_color='#9429e3')
        st.plotly_chart(fig, use_container_width=True)
    
    elif vis_type == "Scatter plot: y_pred vs y_test (log scale)":
        fig = px.scatter(x=data_test["log_price"], y=data_test["log_price_predicted"], trendline="lowess")
        fig.update_layout(xaxis_title="log10(Price)", yaxis_title="Predict")
        fig.data[1].update(line_color='#9429e3')
        st.plotly_chart(fig, use_container_width=True)
    
    elif vis_type == "Scatter plot: percentage error (natural scale)":
        fig = px.scatter(x=data_test["price"], y=data_test["percentage_error"], trendline="lowess")
        fig.update_layout(xaxis_title="Price (R$)", yaxis_title="Percentage error (%)")
        fig.data[1].update(line_color='#9429e3')
        st.plotly_chart(fig, use_container_width=True)
    
    elif vis_type == "Scatter plot: percentage error (logarithmic scale)":
        fig = px.scatter(x=data_test["log_price"], y=data_test["log_percentage_error"], trendline="lowess")
        fig.update_layout(xaxis_title="log10(Price)", yaxis_title="Percentage error (%)")
        fig.data[1].update(line_color='#9429e3')
        st.plotly_chart(fig, use_container_width=True)
    