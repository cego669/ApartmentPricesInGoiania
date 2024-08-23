# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:05:26 2024

@author: cego
"""

from imgs_and_main_page import get_imgs_and_main_page
from apartment_info import get_apartment_info
from apartment_img import get_apartment_img
import dill
import time
import pandas as pd
import traceback

# TIME_LIMIT
TIME_LIMIT = 7*24*60*60 # 7 days at maximum

# links and image extraction (it can take 4 hours)
links = get_imgs_and_main_page()

# saving the dictionary of info
with open("links_dict.pk", "wb") as file:
    dill.dump(links, file)

# dataset extraction, limited by TIME_LIMIT
data = []
before = time.time()
for ID, main_page, img in zip(links["id"],
                              links["main_page"],
                              links["img"]):
    
    # TIME_LIMIT
    if time.time() - before > TIME_LIMIT:
        break
    
    try:
        data.append(get_apartment_info(ID, main_page))
        get_apartment_img(ID, img)
    except Exception as error:
        print(f"ID: {ID},\nError: {error}")
        traceback.print_exc()

# saving dataframe to a .csv file
data = pd.DataFrame(data)
data.to_csv("data.csv",
            index=False,
            sep="|")