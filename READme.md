**INFO240 DATA ANALYTICS**

*GROUP 6*
*JAMES JAMES R206812N*
*ACHIM DANDA R2010022V*
*GERALD PADZINOENDA R2010096E*
*KNOWLEDGE M. CHAWUKURA R202687T*
*SIKHANYISIWE MAJONI R2012095C*

*Assignment 3 - Create a topic model based on the following tweets from the data sets provided*


PYTHON ENVIRONMENT
------------------
Enter tweetModel directory and run the following command to install pipenv module to create an standalone virtual environment::
    $ pip install pipenv
    $ pip install notebook

Create virtual environment::    
    $ pipenv shell

Install all main dependencies::   
    $ pipenv install

To build and test the model run::
    $ py twtModel.py

ALTERNATIVE CODE EXECUTIONS
---------------------------

To run using Jupyter Notebook Environment, move to /tweetModel directory and run::
    $ jupyter notebook 

To run in Python Interactive Shell you can change the third parameter to number of your choice::
    $ python
    $ from twtModel import *
    $ print(display_topics(saved_lda_model, tf_feature_names, 10))




