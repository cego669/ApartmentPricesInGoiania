# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:36:56 2024

@author: cego
"""

from selenium import webdriver
import time

def get_apartment_info(ID, main_page):
    
    # chrome webdriver
    chrome = webdriver.Chrome()
    
    # apartments info
    info = {"id": ID}
    
    # accessing main page of the apartment
    chrome.get(main_page)
    chrome.maximize_window()
    
    # wait 3 seconds until the page is fully loaded
    time.sleep(3)
    
    # tag to know if its "Em construção" (building stage)
    try:
        info["tag_card"] = chrome.\
            find_element("class name",
                         "details-content__info-tags").\
                text
    except:
        pass
    
    # business type
    info["business_type"] = chrome.\
        find_element("id",
                     "business-type-info").\
            text
    
    # apartment price
    info["price"] = chrome.\
        find_element("class name",
                     "l-text l-u-color-neutral-28 l-text--variant-display-regular l-text--weight-bold price-info-value".replace(" ", ".")).\
            text
    
    # condominium fee
    try:
        info["condo_fee"] = chrome.\
            find_element("id",
                         'condo-fee-price').\
                text
    except:
        pass
    
    # IPTU price
    try:
        info["iptu"] = chrome.\
            find_element("id",
                         'iptu-price').\
                text
    except:
        pass
    
    # apartment address
    info["address"] = chrome.\
        find_element("class name",
                     "l-text l-u-color-neutral-28 l-text--variant-body-regular l-text--weight-bold address-info-value".replace(" ", ".")).\
            text
    
    # scroll to "Todas as características" (all characteristics) and give a click (if the button is found)
    try:
        all_characteristics_button = chrome.\
            find_element("class name",
                         "l-link l-link--context-primary l-link--weight-regular l-collapse__action-component".replace(" ", "."))
        chrome.\
            execute_script("arguments[0].scrollIntoView();",
                           all_characteristics_button)
        all_characteristics_button.click()
    except:
        pass
    
    # identify the complete characteristics list
    amenities_list = chrome.\
        find_element("class name",
                     "amenities-list")
    
    # identify the items in the list
    items_in_amenities_list = amenities_list.\
        find_elements("class name",
                      "l-text l-u-color-neutral-28 l-text--variant-body-small l-text--weight-regular amenities-item".replace(" ", "."))
    
    # for each item, store it together with the respective text in the info dict
    for item_amenity in items_in_amenities_list:
        info[item_amenity.get_attribute("itemprop")] =\
            item_amenity.find_element("class name", "amenities-item-text").text
    
    # scroll to the "Descrição completa" (complete description) and give a click (if the button is found)
    try:
        complete_description_button = chrome.\
            find_element("class name",
                         "collapse-toggle-button")
        chrome.\
            execute_script("arguments[0].scrollIntoView();",
                           complete_description_button)
        complete_description_button.click()
    except:
        pass
    
    # complete apartment description
    info["complete_description"] = chrome.\
        find_element("class name",
                     "description__content--text").\
            text
    
    chrome.close()
    
    return info