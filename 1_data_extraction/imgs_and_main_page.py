# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:30:32 2024

@author: cego
"""

from selenium import webdriver
import uuid
from selenium.common.exceptions import StaleElementReferenceException,\
    ElementClickInterceptedException
import time

def get_imgs_and_main_page(time_limit=9999999999):
    
    # webpage with lot of apartments in Goiânia, Goiás
    website = "https://www.vivareal.com.br/venda/goias/goiania/apartamento_residencial/#onde=,Goi%C3%A1s,Goi%C3%A2nia,,,,,city,BR%3EGoias%3ENULL%3EGoiania,-16.686891,-49.264794,&itl_id=1000183&itl_name=vivareal_-_botao-cta_buscar_to_vivareal_resultado-pesquisa"
    
    # chrome webdriver
    chrome = webdriver.Chrome()
    
    # for each card, extract the image link and the main webpage for the apartment
    links = {"id": [],
             "img": [],
             "main_page": []}
    
    # accessing the website
    chrome.get(website)
    chrome.maximize_window()
    
    # time for reference
    before = time.time()
    now = time.time()
    
    # while the time limit isn't reached...
    while (now - before) < time_limit:
        
        # wait 15 seconds
        time.sleep(15)
        
        try:
            # find the main panel
            results_main_panel = chrome.find_element("class name",
                                                     "results-main__panel js-list".replace(" ", "."))
            
            # find the list of apartment cards
            property_card_container = results_main_panel.find_elements("class name",
                                                                       "property-card__container js-property-card".replace(" ", "."))
            
            # for each apartment in the property card container...
            for property_card in property_card_container:
                
                    # scroll to the apartament
                    chrome.\
                        execute_script("arguments[0].scrollIntoView();", property_card)
                    
                    # waite 0.3 second to load the image
                    time.sleep(0.3)
                    
                    img = property_card.\
                                     find_element("tag name", "img").\
                                     get_attribute("src")
                    
                    main_page = property_card.\
                                     find_element("tag name", "a").\
                                     get_attribute("href")
                
                    # take the data
                    links["id"].append(uuid.uuid4().int)
                    links["img"].append(img)
                    links["main_page"].append(main_page)
            
            try:
                # click on "Próxima Página" (next page button)
                next_page_button = results_main_panel.\
                    find_element("xpath", '//*[@title="Próxima página"]')
                chrome.\
                    execute_script("arguments[0].scrollIntoView();",
                                   next_page_button)
                next_page_button.click()
                
            except ElementClickInterceptedException as error:
                # if it isn't possible, then its time to stop
                print(error)
                print("Total rows collected:", len(links["img"]))
                # break the while
                break
        
        except StaleElementReferenceException as error:
            print(error)
            chrome.refresh()
        
        # calculate new "now"
        now= time.time()
    
    chrome.quit()
    
    return links