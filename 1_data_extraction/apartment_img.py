# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:46:53 2024

@author: cego
"""

import requests

def get_apartment_img(ID, img):
    
    # downloading image and saving it to a folder
    img_data = requests.get(img).content
    
    with open(f'imgs/{ID}.png', 'wb') as image:
        image.write(img_data)
