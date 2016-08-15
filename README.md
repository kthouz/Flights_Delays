## Flights Delays

### 1. Objectives: 
1. Build a model to predict how long a flight is to be delayed given flight date, origin and destionation airports as well as the airline
2. Build a user friendly web application

### 2. Data: 
The following data are first collected in database

- [Flight delays data from transtats.bts.gov] (http://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time) 

- [Airports statistics from faa.gov] (http://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/previous_years/)

### 3. Tools:
- Python
- SQL
- HTML, CSS, Javascript, D3JS
### 4. Strategy:
The dataset contains about 30 features about a flight and the airports. However, at the time of booking, the only information available to a user (passenger) are date, origin, destination and airline. In order to take advantage all of the information provided by our data sources, I first analyze all features about airports separately from features about a flights and then generate clusters of similar airports or flights.

1. [Airports analysis] (https://github.com/kthouz/Flights_Delays/blob/master/01_1_Airports_Analysis.ipynb): run statistical analysis and cluster similar aiports together

2. [Flights analysis] (https://github.com/kthouz/Flights_Delays/blob/master/01_2_Flights_Analysis.ipynb): run statistics and cluster similar flights together

3. Delays analysis: build predictive model

### 5. Application:
Build a front-end of a web application using HTML, CSS, JavaScript and D3JS
