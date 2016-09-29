# This scripts use flask framework to run the application 
from flask import Flask, render_template, request, jsonify, json
import aa
import datetime as dt

meta_data = aa.Meta_data()
major_airports = meta_data.major_airports
major_airlines = meta_data.major_airlines
origin_summary = meta_data.origin_summary
dest_summary = meta_data.dest_summary

#it appears that although HA is among major airlines, doesnot have any flight at the major airports, so we remove it
major_airlines.pop('HA')

predictors = aa.Predictors()

app = Flask(__name__)
def __ap_name_acronym__(name):
	ap_name_ac = {}
	for k,v in dict(major_airports.HUB_NM).iteritems():
		ap_name_ac[v] = k
	return ap_name_ac[name]

def __al_name_acronym__(name):
	al_name_ac = {}
	for k,v in major_airlines.iteritems():
		al_name_ac[v] = k
	return al_name_ac[name]

def get_origin_summary(airport_code,year,month):
	rank = None
	mean_delay = None
	std_delay = None
	del_rate = None
	cancel_rate = None

def get_destination_summary(airport_code,year,month):
	rank = None
	mean_delay = None
	std_delay = None
	del_rate = None
	divert_rate = None


@app.route('/get_airports')
def get_airports():
	major_airports_nm = major_airports.HUB_NM.tolist()
	return sorted(major_airports_nm)

@app.route('/get_airlines')
def get_airlines():
	major_airlines_nm = major_airlines.values()

	return sorted(major_airlines_nm)

@app.route('/make_predictions')
def make_predictions():
	variables = request.args.get('variables',0,str)
	# parse data
	data = variables.split('__')
	origin = __ap_name_acronym__(data[0])
	destination = __ap_name_acronym__(data[1])
	airline = __al_name_acronym__(data[2])
	date = map(lambda x:int(x),data[3].split('/'))
	
	week = dt.date(date[2],date[0],date[1]).isocalendar()[1]
	
	variables = {'airline':airline,'from':origin,'to':destination,'week':week}
	probability = predictors.predict(variables)

	ostats = meta_data.summarize_origin(origin,airline,date[2]-1,week)
	ostats_html = "Departure delay rate: {0}%<br/>Cancellation rate: {1}%<br/>Delay time: {2} &plusmn {3}min<br/>Number out flights (same week {5}): {4}".format(ostats['delay_rate'],ostats['cd_rate'],ostats['mean_delay_time'],ostats['std_delay_time'],ostats['nflights'],date[2]-1)
	dstats = meta_data.summarize_destination(destination,airline,date[2]-1,week)	
	dstats_html = "Arrival delay rate: {0}%<br/>Divert rate: {1}%<br/>Delay time: {2} &plusmn {3}min<br/>Number in flights (same week {5}): {4}".format(dstats['delay_rate'],dstats['cd_rate'],dstats['mean_delay_time'],dstats['std_delay_time'],dstats['nflights'],date[2]-1)

	flstats = meta_data.summarize_flight(origin,destination,airline,date[2]-1,week)
	flstats['stats'] = "Number flights: "+flstats['nflights']+"<br/>"+origin+" departure delay time: "+flstats['mean_ddelay_time']+" &plusmn "+flstats['std_ddelay_time']+"min<br/>"
	flstats['stats'] += destination+" arrival delay time: "+flstats['mean_adelay_time']+" &plusmn "+flstats['std_adelay_time']+"min<br/>"+origin+" departure delay rate: "+flstats['ddel_rate']+"%<br/>"
	flstats['stats'] += destination+" arrival delay rate: "+flstats['adel_rate']+"%<br/>"

	print variables, probability,flstats['stats']
	return jsonify(status='received', 
		probability=probability, 
		origin={'hub':origin,'lon':major_airports.loc[origin,'LON'],'lat':major_airports.loc[origin,'LAT'],'stats':ostats_html}, 
		destination={'hub':destination,'lon':major_airports.loc[destination,'LON'],'lat':major_airports.loc[destination,'LAT'],'stats':dstats_html}, 
		flight = flstats, 
		date=data[3])

@app.route('/')
def index():
	airports = sorted(major_airports.HUB_NM.tolist())#get_airports()
	airlines = sorted(major_airlines.values())#get_airlines()
	return render_template('index.html',airports=airports, airlines=airlines)

if __name__ == '__main__':
    app.run(debug=True)
