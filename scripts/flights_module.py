import pandas as pd
import numpy as np
import datetime as dt
#import matplotlib.pyplot as plt
import json, zipfile, StringIO, requests

import warnings
warnings.filterwarnings('ignore')

# initialize a connection to the database
import sqlite3
conn = sqlite3.connect('data/database.db')

# initialize another sqlalchemy connection to the same database to be able to query data straight to pandas dataframe
from sqlalchemy import create_engine
disk_engine = create_engine('sqlite:///data/database.db')

class Transtats():
    def __init__(self):
        self.__lookup_Airports__()
        self.__lookup_CancellationCodes__()
        self.__lookup_Carriers__()
        self.__lookup_DelayGroups__()
        self.__lookup_TimeBlocks__()
        self.__lookup_DistanceGroups__()
    
    def __lookup_Airports__(self):
        tbl = 'lookup_Airport'
        tables_list = map(lambda x:x[0],conn.execute("select name from sqlite_master where type = 'table'").fetchall())
        if tbl in tables_list:
            self.airports_list = pd.read_sql_query("SELECT * FROM {0}".format(tbl),disk_engine)
        else:
            link = "http://www.transtats.bts.gov/Download_Lookup.asp?Lookup=L_AIRPORT"
            df = pd.read_csv(link)
            airports = pd.DataFrame(index=df.Code.values,columns=['DESCRIPTION'])
            airports.DESCRIPTION = df.Description.apply(lambda x:x.split(',')[0]).values.tolist()
            self.airports_list = airports
            self.airports_list.to_sql(tbl,conn)
    
    def __lookup_CancellationCodes__(self):
        tbl = 'lookup_CancellationCode'
        tables_list = map(lambda x:x[0],conn.execute("select name from sqlite_master where type = 'table'").fetchall())
        if tbl in tables_list:
            self.cancellation_codes = pd.read_sql_query("SELECT * FROM {0}".format(tbl),disk_engine)
        else:
            link = "http://www.transtats.bts.gov/Download_Lookup.asp?Lookup=L_CANCELLATION"
            df = pd.read_csv(link)
            self.cancellation_codes = pd.DataFrame(index = df.Code.values, columns = ['DESCRIPTION'])
            self.cancellation_codes.DESCRIPTION = df.Description.values
            self.cancellation_codes.to_sql(tbl,conn)
            
    
    def __lookup_Carriers__(self):
        tbl = 'lookup_Carrier'
        tables_list = map(lambda x:x[0],conn.execute("select name from sqlite_master where type = 'table'").fetchall())
        if tbl in tables_list:
            self.carriers = pd.read_sql_query("SELECT * FROM {0}".format(tbl),disk_engine)
        else:
            link = "http://www.transtats.bts.gov/Download_Lookup.asp?Lookup=L_CARRIER_HISTORY"
            df = pd.read_csv(link)
            self.carriers = pd.DataFrame()
            self.carriers['CODE'] = df.Code.values
            self.carriers['DESCRIPTION'] = df.Description.apply(lambda x:x.split('(')[0]).values
            self.carriers.to_sql(tbl,conn,index=False)
    
    def __lookup_DelayGroups__(self):
        tbl = 'lookup_DelayGroup'
        tables_list = map(lambda x:x[0],conn.execute("select name from sqlite_master where type = 'table'").fetchall())
        if tbl in tables_list:
            self.delay_groups = pd.read_sql_query("SELECT * FROM {0}".format(tbl),disk_engine)
        else:
            link = "http://www.transtats.bts.gov/Download_Lookup.asp?Lookup=L_ONTIME_DELAY_GROUPS"
            df = pd.read_csv(link)
            self.delay_groups = pd.DataFrame(index = df.Code.values)
            self.delay_groups['DESCRIPTION'] = df.Description.values
            self.delay_groups.to_sql(tbl,conn)
    
    def __lookup_TimeBlocks__(self):
        tbl = 'lookup_TimeBlock'
        tables_list = map(lambda x:x[0],conn.execute("select name from sqlite_master where type = 'table'").fetchall())
        if tbl in tables_list:
            self.time_blocks = pd.read_sql_query("SELECT * FROM {0}".format(tbl),disk_engine)
        else:
            link = "http://www.transtats.bts.gov/Download_Lookup.asp?Lookup=L_DEPARRBLK"
            df = pd.read_csv(link)
            self.time_blocks = pd.DataFrame(index = df.Code.values)
            self.time_blocks['DESCRIPTION'] = df.Description.values
            self.time_blocks.to_sql(tbl,conn)
        
    def __lookup_DistanceGroups__(self):
        tbl = 'lookup_DistanceGroup'
        tables_list = map(lambda x:x[0],conn.execute("select name from sqlite_master where type = 'table'").fetchall())
        if tbl in tables_list:
            self.distance_groups = pd.read_sql_query("SELECT * FROM {0}".format(tbl),disk_engine)
        else:
            link = "http://www.transtats.bts.gov/Download_Lookup.asp?Lookup=L_DISTANCE_GROUP_250"
            df = pd.read_csv(link)
            self.distance_groups = pd.DataFrame(index=df.Code.values)
            self.distance_groups['DESCRIPTION'] = df.Description.values
            self.distance_groups.to_sql(tbl,conn)
    
    def get_flightsData(self,year,month):
        #print year, month,"...",
        columns = [u'Year', u'Quarter', u'Month', u'DayofMonth', u'DayOfWeek',u'FlightDate', u'UniqueCarrier', 
                   u'AirlineID', u'Carrier', u'TailNum', u'FlightNum', u'OriginCityMarketID', u'Origin', 
                   u'OriginCityName', u'OriginState', u'DestCityMarketID', u'Dest', u'DestCityName','DestState', 
                   u'CRSDepTime', u'DepTime', u'DepDelay', u'DepDelayMinutes', u'DepDel15', u'DepartureDelayGroups', 
                   u'DepTimeBlk', u'TaxiOut', u'TaxiIn', u'CRSArrTime', u'ArrTime', u'ArrDelay', u'ArrDelayMinutes', 
                   u'ArrDel15', u'ArrivalDelayGroups', u'ArrTimeBlk', u'Cancelled', u'CancellationCode', u'Diverted', 
                   u'CRSElapsedTime', u'ActualElapsedTime', u'AirTime', u'Flights', u'Distance', u'DistanceGroup', 
                   u'CarrierDelay', u'WeatherDelay', u'NASDelay', u'SecurityDelay', u'LateAircraftDelay']
        #tic = dt.datetime.now()
        try:
            r = requests.get('http://tsdata.bts.gov/PREZIP/On_Time_On_Time_Performance_{0}_{1}.zip'.format(year,month))
            z = zipfile.ZipFile(StringIO.StringIO(r.content))
            df = pd.read_csv(z.open('On_Time_On_Time_Performance_{0}_{1}.csv'.format(year,month)))[columns]
            #print "Done! ", dt.datetime.now()-tic
            return df
        except:
            print "error and skipped"
            df = pd.DataFrame(columns=columns)
            return df

class Faa():
    def __init__(self):
        self.__get_airportRanking__()
        self.__get_largeHubs__()
        self.__get_airportCoordinates__()
    
    def __get_airportRanking__(self):
        tbl = 'Airport_Rank'
        tables_list = map(lambda x:x[0],conn.execute("select name from sqlite_master where type = 'table'").fetchall())
        if tbl in tables_list:
            self.airports_ranks = pd.read_sql_query("SELECT * FROM {0}".format(tbl),disk_engine)
        else:
            # get airports ranking from FAA
            links = [[2015,"http://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/media/preliminary-cy15-commercial-service-enplanements.xlsx"],
                     [2014,"http://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/media/cy14-commercial-service-enplanements.xlsx"],
                     [2013,"http://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/media/cy13-commercial-service-enplanements.xlsx"],
                     [2012,"http://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/media/CY12CommercialServiceEnplanements.xlsx"],
                     [2011,"http://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/media/cy11_primary_enplanements.xlsx"],
                     [2010,"http://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/media/cy10_primary_enplanements.xls"]]

            ranks = pd.DataFrame()
            for link in sorted(links):
                df = pd.read_excel(link[1])
                df.rename(columns={df.columns[-2]:"Enplanement",df.columns[-1]:"Change"}, inplace=True)
                df['Year'] = link[0]
                ranks = pd.concat([ranks,df[['Year','Rank','Locid','Hub','Enplanement','Change']]],axis=0,ignore_index=True)
            # drop rows which do not correspond to airports data and reset indices
            indices_to_drop = ranks[ranks.Locid.isnull()].index.tolist()
            ranks.drop(indices_to_drop,inplace=True)
            ranks = ranks.reset_index()
            ranks.rename(columns = {'Locid':'Iata'}, inplace=True)
            ranks.drop('index',axis=1,inplace=True)
            self.airports_ranks = ranks
            self.airports_ranks.to_sql(tbl,conn,index=False)
            
    def __get_largeHubs__(self):
        ranks = self.airports_ranks.copy()
        self.large_hubs = map(lambda x:str(x),pd.unique(ranks[ranks.Hub == "L"]["Iata"]).tolist())
    
    def __get_airportCoordinates__(self):
        tbl = 'Airport_Coordinates'
        tables_list = map(lambda x:x[0],conn.execute("select name from sqlite_master where type = 'table'").fetchall())
        if tbl in tables_list:
            self.airports_coordinates = pd.read_sql_query("SELECT * FROM {0}".format(tbl),disk_engine)
        else:
            link = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
            df = pd.read_csv(link,header=None)[[1,3,4,6,7,8,9]]
            df = df[df[3] == "United States"].reset_index() #selected airports in USA
            df.drop([3,'index'],axis=1,inplace=True)
            df.columns = ["Airport","Iata","Latitude","Longitude","Altitude","Timezone"]
            self.airports_coordinates = df.copy()
            self.airports_coordinates.to_sql(tbl,conn,index=False)