#!/usr/bin/env python
# coding: utf-8

# This exercise will require you to pull some data from https://data.nasdaq.com/ (formerly Quandl API).

# As a first step, you will need to register a free account on the https://data.nasdaq.com/ website.

# After you register, you will be provided with a unique API key, that you should store:
# 
# *Note*: Use a `.env` file and put your key in there and `python-dotenv` to access it in this notebook. 
# 
# The code below uses a key that was used when generating this project but has since been deleted. Never submit your keys to source control. There is a `.env-example` file in this repository to illusrtate what you need. Copy that to a file called `.env` and use your own api key in that `.env` file. Make sure you also have a `.gitignore` file with a line for `.env` added to it. 
# 
# The standard Python gitignore is [here](https://github.com/github/gitignore/blob/master/Python.gitignore) you can just copy that. 

# In[1]:


python--version


# In[7]:


# get api key from your .env file
import os
import pip
get_ipython().system('pip install python-dotenv')
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY_Nasdaq2')
print(API_KEY)


# Nasdaq Data has a large number of data sources, but, unfortunately, most of them require a Premium subscription. Still, there are also a good number of free datasets.

# For this mini project, we will focus on equities data from the Frankfurt Stock Exhange (FSE), which is available for free. We'll try and analyze the stock prices of a company called Carl Zeiss Meditec, which manufactures tools for eye examinations, as well as medical lasers for laser eye surgery: https://www.zeiss.com/meditec/int/home.html. The company is listed under the stock ticker AFX_X.

# You can find the detailed Nasdaq Data API instructions here: https://docs.data.nasdaq.com/docs/in-depth-usage

# While there is a dedicated Python package for connecting to the Nasdaq API, we would prefer that you use the *requests* package, which can be easily downloaded using *pip* or *conda*. You can find the documentation for the package here: http://docs.python-requests.org/en/master/ 

# Finally, apart from the *requests* package, you are encouraged to not use any third party Python packages, such as *pandas*, and instead focus on what's available in the Python Standard Library (the *collections* module might come in handy: https://pymotw.com/3/collections/).
# Also, since you won't have access to DataFrames, you are encouraged to us Python's native data structures - preferably dictionaries, though some questions can also be answered using lists.
# You can read more on these data structures here: https://docs.python.org/3/tutorial/datastructures.html

# Keep in mind that the JSON responses you will be getting from the API map almost one-to-one to Python's dictionaries. Unfortunately, they can be very nested, so make sure you read up on indexing dictionaries in the documentation provided above.

# In[2]:


# First, import the relevant modules
import requests


# Note: API's can change a bit with each version, for this exercise it is reccomended to use the nasdaq api at `https://data.nasdaq.com/api/v3/`. This is the same api as what used to be quandl so `https://www.quandl.com/api/v3/` should work too.
# 
# Hint: We are looking for the `AFX_X` data on the `datasets/FSE/` dataset.

# In[10]:


# Now, call the Nasdaq API and pull out a small sample of the data (only one day) to get a glimpse
# into the JSON structure that will be returned
api_url = 'https://data.nasdaq.com/api/v3/datasets/WIKI/AAPL.csv'
response = requests.get(api_url)
response.status_code
response.json()


# In[4]:


# Inspect the JSON structure of the object you created, and take note of how nested it is,
# as well as the overall structure
import json


# These are your tasks for this mini project:
# 
# 1. Collect data from the Franfurt Stock Exchange, for the ticker AFX_X, for the whole year 2017 (keep in mind that the date format is YYYY-MM-DD).
# 2. Convert the returned JSON object into a Python dictionary.
# 3. Calculate what the highest and lowest opening prices were for the stock in this period.
# 4. What was the largest change in any one day (based on High and Low price)?
# 5. What was the largest change between any two days (based on Closing Price)?
# 6. What was the average daily trading volume during this year?
# 7. (Optional) What was the median trading volume during this year. (Note: you may need to implement your own function for calculating the median.)

# In[ ]:





# In[ ]:




