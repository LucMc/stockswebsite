# Falco Stcok Analysis
Code submitted in partial fulfilment of the requirements for the award of Batchelor in Computer Science.

The majority of the code can be found in `stockswebsite/main/strategy_comparison/Stocks/analytics`


### Requirements
 - Python 3.8
 - Windows 10

This code was ran and tested for Windows. It should work on MAC (not on x86 architecture) and Linux however one may need to install requirements.


### Installation
To install type the `pip install -r requirements.txt` whilst in the stockswebsite directory.

### Run
To run the code use `python3 manage.py runserver`

### Usage
Type in the ticker for the security you would like to assess in the top right, select a year of inquiry and then Search.
The program can take a number of seconds to generate the required graphs before serving the visualisation.

Note the tickers should be entered as found on 
https://uk.finance.yahoo.com/quote/GBPUSD=X/

For the docker image please find instructions:
https://docs.docker.com/language/nodejs/run-containers/

1. Navigate into the stockswebsite folder where the Dockerfile is located

2. `docker build --tag falco:latest .`
   
3. `docker images`
   Find your image id
   
4. `docker run --name falco -d -p 8000:8000 [input image id here]`

5. go to your internet browser and visit `localhost:8000`



For the hosted webapp visit (not recommended):
https://falco-stock.herokuapp.com