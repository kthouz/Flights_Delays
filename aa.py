import pandas as pd
import numpy as np
import datetime as dt
import colorlover as cl
import matplotlib.pyplot as plt
from matplotlib import cm as cmap
import glob, os, re, pickle, sqlite3#, pyspark
from IPython.display import Image, HTML
from sqlalchemy import create_engine
#from pyspark.sql import SQLContext
#sqlctx = SQLContext(sc)

import plotly 
import plotly.tools as tls
import plotly.graph_objs as go
ply_credentials = pd.read_csv('../Spotify_Challenge/plotly_credentials.csv')
plotly.tools.set_credentials_file(username=ply_credentials.values[0][0], api_key=ply_credentials.values[0][1])
#%matplotlib inline

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import scipy.optimize as spo

class DB():
    def __init__(self):
        """
        db object to push new data to the data base
        """
        # initialize connection
        self.conn = sqlite3.connect('../Spotify_Challenge/dbsql.db')
        self.__get_tables_names__()

    def __get_tables_names__(self):
        cmd = "select name from sqlite_master where type = 'table'"
        r = self.conn.execute(cmd)
        self.list_tables = map(lambda x:str(x[0]),r.fetchall())

    def create_table(self,table_name,attributes):
        """
        table_name: str
        attributes: list of attributes
        """
        try:
            #check if table doesn't exist already and create table
            if not table_name in self.list_tables:
                cmd = "CREATE TABLE {0}(".format(table_name)
                for k in attributes:
                    cmd+="{0}, ".format(k)
                cmd = cmd[:-2]+")"
                self.conn.execute(cmd)

                #update list of tables
                self.__get_tables_names__()
                flag = 1
            else:
                print "table exists already"
                flag = 0
        except:
            print "Table note created"
            print "issued command: "+cmd
            flag = 0
        return flag


    def drop_table(self,table_name):
        try:
            #check if table exist already and drop it
            if table_name in self.list_tables:
                cmd = "DROP TABLE {0}".format(table_name)
                self.conn.execute(cmd)
                #update list of tables
                self.__get_tables_names__()
                flag = 1
            else:
                print "Table does not exist"
                flag = 0
        except:
            print "Table note created"
            print "issued command: "+cmd
            flag = 0
        return flag

    def push_df(self,df,table_name,index=False):
        """
        push pandas dataframe to db
        """
        try:
            if table_name in self.list_tables:
                self.drop_table(table_name)
            df.to_sql(table_name,self.conn,index=index)
            flag = 1
        except:
            print "dataframe to pushed"
            flag = 0
        return flag

    def close_connection(self):
        self.conn.close()

class Meta_data():
    def __init__(self):
        # initiate a connection to the database
        tic = dt.datetime.now()
        self.disk_engine = create_engine('sqlite:///../Spotify_Challenge/dbsql.db')
        self.cols = [u'YEAR', u'QUARTER', u'MONTH', u'DAY_OF_MONTH', u'DAY_OF_WEEK', u'AIRLINE_ID', u'CARRIER',u'TAIL_NUM', 
                u'FL_NUM', u'ORIGIN_AIRPORT_ID',u'ORIGIN', u'ORIGIN_CITY_NAME', u'ORIGIN_STATE_NM', u'DEST_AIRPORT_ID', 
                u'DEST', u'DEST_CITY_NAME', u'DEST_STATE_NM', u'DEP_DELAY', u'DEP_DELAY_NEW', u'DEP_DEL15', u'TAXI_OUT',
                u'TAXI_IN', u'ARR_DELAY', u'ARR_DELAY_NEW', u'ARR_DEL15', u'CANCELLED', u'CANCELLATION_CODE', u'DIVERTED', 
                u'AIR_TIME', u'FLIGHTS', u'DISTANCE',u'CARRIER_DELAY', u'WEATHER_DELAY', u'NAS_DELAY', u'SECURITY_DELAY', 
                u'LATE_AIRCRAFT_DELAY', u'FL_DATE']
        
        self.major_airlines = self.__get_major_airlines__()
        self.major_airports = self.__get_major_airports__(limit=20)
        self.origin_summary = self.__get_origin_summary__()
        self.dest_summary = self.__get_destination_summary__()
        
        print "meta_data collected in ", dt.datetime.now()-tic
    
    def __get_major_airlines__(self):
        """
        returns dictionary of airlines that were operational since 2010
        """
        if os.path.exists('major_airlines.csv'):
            print "loading major carriers from disc ..."
            major_carriers = pd.read_csv('major_airlines.csv',index_col='CARRIER')
            return major_carriers.to_dict()['CARRIER_NM']
        else:
            print "quering major carriers from the database. Just hang on, it may take few minutes ..."
            # query carriers by yearly performance
            df_major_carriers = pd.read_sql_query(
                'SELECT CARRIER, YEAR, COUNT(CARRIER) as CNT '
                'FROM data '
                'GROUP BY YEAR,CARRIER',
                 self.disk_engine)

            # transform dataframe from long to wide with CARRIER as index and YEAR as columns. For an unoperational CARRIER C
            # during YEAR Y, the corresponding cell (C,Y) will be NaN. By droping NaN, 
            # we will stay with major carrier as to the definition
            df_major_carriers = pd.pivot(index = df_major_carriers.CARRIER, columns=df_major_carriers.YEAR, 
                                         values = df_major_carriers.CNT).T.dropna(axis=1).reset_index()
            df_major_carriers.index = df_major_carriers.YEAR.values
            df_major_carriers.drop('YEAR',axis=1,inplace=True)
            major_carriers = map(lambda x:str(x),df_major_carriers.columns.tolist())

            # get full names of all carriers
            df_carriers = pd.read_sql_query('SELECT DISTINCT carriers.Code as CARRIER, carriers.Description as CARRIER_NM '
                                 'FROM carriers '
                                 'WHERE CARRIER IN {0}'.format(tuple(major_carriers)),
                                  self.disk_engine)
            df_carriers.index = df_carriers.CARRIER
            df_carriers.drop('CARRIER', axis=1, inplace=True)
            df_carriers.to_csv('major_airlines.csv')
            
            return df_carriers.to_dict()['CARRIER_NM']
    
    def __get_major_airports__(self,limit=20):
        """
        returns any airport that was among top 20 airports to host many flights since 2010
        """
        if os.path.exists('major_airports.csv'):
            print "loading major airports from disc ..."
            major_airports = pd.read_csv('major_airports.csv',index_col='Unnamed: 0')
            return major_airports
        else:
            print "quering major airports from the database. Just hang on, it may take few minutes ..."
            dc_major_airports = {}
            years = [2010,2011,2012,2013,2014,2015,2016]
            for year in years:
                tic = dt.datetime.now()
                df = pd.read_sql_query('SELECT DISTINCT ORIGIN, COUNT(ORIGIN) as CNT '
                                       'FROM data '
                                       'WHERE YEAR = {0} '
                                       'GROUP BY ORIGIN '
                                       'ORDER BY CNT DESC '
                                       'LIMIT {1}'.format(year,limit),
                                       self.disk_engine)
                dc_major_airports[year] = pd.Series(index=df.ORIGIN.tolist(), data=df.CNT.tolist())
                print year,"querry done in ", dt.datetime.now()-tic

            # create a list of major airports
            major_airports = []
            for k,v in dc_major_airports.iteritems():
                major_airports = major_airports+v.index.tolist()
            major_airports=tuple(map(lambda x:str(x),set(major_airports)))

            # build airport dataframe with geo information
            # get all airports in data and join on airports
            df_airports = pd.read_sql_query('SELECT DISTINCT iata as CODE, airport as HUB_NM, '
                                            'city as CITY, state as STATE, lat as LAT, long as LON '
                                            'FROM airports '
                                            'WHERE CODE IN {0}'.format(major_airports),
                                             self.disk_engine)
            df_airports.index = df_airports.CODE.values
            df_airports.drop('CODE', axis=1,inplace=True)
            df_airports.to_csv('major_airports.csv')
            major_airports = df_airports.copy()
            
            return major_airports

    def __get_origin_summary__(self):
        if os.path.exists('origin_summary.csv'):#check if it file exists on disk already
            print "loading origin summary from disk"
            df = pd.read_csv('origin_summary.csv')
            return df
        else:#query the db if the file is not found on disk
            print "querying origin summary from db"
            df = pd.read_sql_query('SELECT ORIGIN, YEAR, MONTH, FLIGHTS, DEP_DELAY_NEW as DELAY, DEP_DEL15 as DEL15, CANCELLED '
                                   'FROM data '
                                   'WHERE ORIGIN IN {0}'.format(tuple(self.major_airports.index.tolist())),self.disk_engine)
            df = df.groupby(['ORIGIN','YEAR','MONTH']).sum().reset_index()
            df.to_csv('origin_summary.csv',index=False)
            return df

    def __get_destination_summary__(self):
        if os.path.exists('destination_summary.csv'):#check if it file exists on disk already
            print "loading destination summary from disk"
            df = pd.read_csv('destination_summary.csv')
            return df
        else:#query the db if the file is not found on disk
            print "querying destination summary from db"
            df = pd.read_sql_query('SELECT DEST, YEAR, MONTH, FLIGHTS, ARR_DELAY_NEW as DELAY, ARR_DEL15 as DEL15, DIVERTED '
                                   'FROM data '
                                   'WHERE DEST IN {0}'.format(tuple(self.major_airports.index.tolist())),self.disk_engine)
            df = df.groupby(['DEST','YEAR','MONTH']).sum().reset_index()
            df.to_csv('destination_summary.csv',index=False)
            return df

    def summarize_origin(self,origin,carrier,year,week):
        df = pd.read_sql_query("select YEAR, WEEK_OF_YEAR, CARRIER, ORIGIN, FLIGHTS, DEP_DEL15 as DEL15, DEP_DELAY_NEW as DELAY, CANCELLED "
                               "from flights "
                               "where YEAR = {0} and WEEK_OF_YEAR = {1} and CARRIER = '{2}' and ORIGIN = '{3}'". format(year,week,carrier,origin),
                               self.disk_engine)
        if df.shape[0]>=1:
            n_flights = df.shape[0]
            delay_rate = int(round(100*df.DEL15.sum()/float(df.shape[0]),0))
            cancellatation_rate = int(round(100*df.CANCELLED.sum()/float(df.shape[0]),0))
            mean_delay_time = int(round(df.DELAY.mean(),0))
            std_delay_time = int(round(df.DELAY.std(),0))
            rank = None
        else:
            n_flights = df.shape[0]
            delay_rate = 100
            cancellatation_rate = 100
            mean_delay_time = 100
            std_delay_time = 100
            rank = None
        return {'nflights':n_flights,'delay_rate':delay_rate,'cd_rate':cancellatation_rate,'mean_delay_time':mean_delay_time,'std_delay_time':std_delay_time,'rank':rank}

    def summarize_destination(self,dest,carrier,year,week):
        df = pd.read_sql_query("select YEAR, WEEK_OF_YEAR, CARRIER, DEST, FLIGHTS, ARR_DEL15 as DEL15, ARR_DELAY_NEW as DELAY, DIVERTED "
                               "from flights "
                               "where YEAR = {0} and WEEK_OF_YEAR = {1} and CARRIER = '{2}' and DEST = '{3}'". format(year,week,carrier,dest),
                               self.disk_engine)
        if df.shape[0]>=1:
            n_flights = df.shape[0]
            delay_rate = int(round(100*df.DEL15.sum()/float(df.shape[0]),0))
            divert_rate = int(round(100*df.DIVERTED.sum()/float(df.shape[0]),0))
            mean_delay_time = int(round(df.DELAY.mean(),0))
            std_delay_time = int(round(df.DELAY.std(),0))
            rank = None
        else:
            n_flights = df.shape[0]
            delay_rate = 100
            divert_rate = 100
            mean_delay_time = 100
            std_delay_time = 100
            rank = None
        return {'nflights':n_flights,'delay_rate':delay_rate,'cd_rate':divert_rate,'mean_delay_time':mean_delay_time,'std_delay_time':std_delay_time,'rank':rank}

    def summarize_flight(self,origin,dest,carrier,year,week):
        df = pd.read_sql_query("select YEAR, WEEK_OF_YEAR, CARRIER, ORIGIN, DEST, FLIGHTS, DEP_DEL15, ARR_DEL15, "
                               "DEP_DELAY_NEW as DDELAY, ARR_DELAY_NEW as ADELAY, DIVERTED, DISTANCE, AIR_TIME "
                               "from flights "
                               "where YEAR = {0} and WEEK_OF_YEAR = {1} and CARRIER = '{2}' "
                               "and ORIGIN = '{3}'  and DEST = '{4}'". format(year,week,carrier,origin,dest),
                               self.disk_engine)
        if df.shape[0]>=1:
            n_flights = str(df.shape[0])
            mean_adelay_time = str(int(round(df.ADELAY.mean())))
            mean_ddelay_time = str(int(round(df.DDELAY.mean())))
            std_adelay_time = str(int(round(df.ADELAY.std())))
            std_ddelay_time = str(int(round(df.DDELAY.std())))
            adel_rate = str(int(round(100*df.ARR_DEL15.sum()/float(df.shape[0]))))
            ddel_rate = str(int(round(100*df.DEP_DEL15.sum()/float(df.shape[0]))))
        else:
            n_flights = str(0)
            mean_adelay_time = 'NA'
            mean_ddelay_time = 'NA'
            std_adelay_time = 'NA'
            std_ddelay_time = 'NA'
            adel_rate = 'NA'
            ddel_rate = 'NA'

        return {'airline':carrier,'nflights':n_flights,'mean_adelay_time':mean_adelay_time,'mean_ddelay_time':mean_ddelay_time,
                'std_adelay_time':std_adelay_time,'std_ddelay_time':std_ddelay_time,'adel_rate':adel_rate,'ddel_rate':ddel_rate}


        return df

    def get_flights(self,dump_to_disk=True):
        """
        This query flights where the airlines is among major airlines and origin and destinations are among major airports
        """
        mydb = DB()
        if 'flights' in mydb.list_tables:
            print "loading previously used and pickled flights data from disc ..."
            df = pd.read_sql_query('SELECT * FROM flights',self.disk_engine)
            mydb.close_connection()

            return df
        else:
            print "querying new data from database. Hang on, it may take few minutes ..."
            tic = dt.datetime.now()
            df_flights = pd.read_sql_query('SELECT YEAR, MONTH, DAY_OF_MONTH, CARRIER, ORIGIN, DEST, FLIGHTS, DEP_DEL15, ARR_DEL15, '
                                           'DEP_DELAY, ARR_DELAY, DEP_DELAY_NEW, ARR_DELAY_NEW, AIR_TIME, DISTANCE, TAXI_IN, TAXI_OUT, '
                                           'CANCELLED, DIVERTED '
                                           'FROM data '
                                           'WHERE CARRIER IN {0} AND ORIGIN IN {1} AND DEST IN {1}'.format(tuple(self.major_airlines.keys()),
                                                             tuple(self.major_airports.index.tolist())), 
                                           self.disk_engine)
            # create column WEEK
            df_flights['WEEK_OF_YEAR'] = df_flights.apply(lambda r: dt.date(r[0],r[1],r[2]).isocalendar()[1],axis=1)
            print "query done in", dt.datetime.now()-tic
            if dump_to_disk:
                #import pickle
                #with open('flights_data.pkl','wb') as df:
                #    pickle.dump(df_flights,df)
                #    df.close()
                
                mydb.push_df(df_flights,'flights',index=False)
            mydb.close_connection()

            return df_flights
    
class Models():
    def __init__(self):
        pass
    
    def __get_data__(self):
        """
        prepare data to use in model training and validation
        """
        self.meta_data = Meta_data()
        self.major_airlines = self.meta_data.major_airlines
        self.major_airports = self.meta_data.major_airports
        self.flights_data = self.meta_data.get_flights()
        
    def __set_parameters__(self,labels = {'train':['CARRIER','ORIGIN','DEST','WEEK_OF_YEAR'],'target':'ARR_DEL15'}):
        """
        labels: dictionary
        """
        self.labels = labels
    
    def __clean_data__(self):
        """
        during the analysis, we have seen that only DEP_DEL15 and ARR_DEL15 contains NaNs.
        Here we are going to directly drop rows with NaN
        Note: if future, be careful before running this line
        """
        print "cleaning data ..."
        print "You are advised to run statistical analysis before droping NaNs"
        self.flights_data.dropna(inplace=True)
        
        # encode string variables to numeric
        
        print "encoding string into numeric variable ..."
        
        self.l_encoder = {}
        
        to_convert = []
        types = self.flights_data.dtypes
        for i in types.index:
            if types.loc[i] == 'object':
                to_convert.append(i)
                self.l_encoder[i] = LabelEncoder()
                self.flights_data[i] = self.l_encoder[i].fit_transform(self.flights_data[i])
                
    def train_models(self,df_train=None,df_valid=None):
        
        print "training the model ..."
        if (df_train == None) or (df_valid == None):
            # split dataset into training and validation set
            #fl_dates = self.flights_data.apply(lambda r: dt.date(int(r[0]),int(r[1]),int(r[2])), axis=1)
            df_train = self.flights_data
            #df_valid = self.flights_data[fl_dates>=dt.date(2015,1,1)]
            df_valid = self.flights_data.copy()
        
        self.gnbmodel = GaussianNB()
        self.gnbmodel.fit(df_train[self.labels['train']],df_train[self.labels['target']])
        delay_probability = self.gnbmodel.predict_proba(df_valid[self.labels['train']])

        # we can set the threshold as the ratio of total delayed flights
        thresh = df_train[self.labels['target']].value_counts(normalize=True)[1]

        plt.hist(delay_probability.transpose()[1], normed=True,color='green')
        plt.axvline(thresh)
        plt.title('Probabilities for a flight to be delayed')
        plt.legend(['thresh','hist'])
        plt.savefig('predicted_probabs.eps', format='eps', dpi = 100)
        plt.show()
        delay_pred = delay_probability.transpose()[1]>=thresh
        print "mse", mean_squared_error(df_valid[self.labels['target']],delay_pred)
        print "acc score", accuracy_score(df_valid[self.labels['target']],delay_pred)
        
    def save_models(self):
        with open('the_model.pkl','wb') as mdl:
            pickle.dump(self.gnbmodel,mdl)
            mdl.close()
        with open('label_encoder.pkl','wb') as le:
            pickle.dump(self.l_encoder,le)
            le.close()
            
class Predictors():
    
    def __init__(self):
        print "loading predictors ..."
        # load the model
        with open('the_model.pkl','r') as mdl:
            self.pred_model = pickle.load(mdl)
            mdl.close()
        with open('label_encoder.pkl','r') as l:
            _encoder = pickle.load(l)
            l.close()
            self.label_encoder = {}
            for k,v in _encoder.iteritems():
                self.label_encoder[k] = {}
                i = 0
                for code in v.classes_:
                    self.label_encoder[k][code] = i
                    i+=1
    
    def predict(self,inputs):
        carrier_num = self.label_encoder['CARRIER'][inputs['airline']]
        origin_num = self.label_encoder['ORIGIN'][inputs['from']]
        dest_num = self.label_encoder['DEST'][inputs['to']]
        week = inputs['week']
        row = np.array([carrier_num,origin_num,dest_num,week])
        print (carrier_num, inputs['airline']),(origin_num,inputs['from']),(dest_num,inputs['to']),week
        return round(self.pred_model.predict_proba(row)[0][1]*100,2)