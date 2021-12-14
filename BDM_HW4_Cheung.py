from pyspark import SparkContext
import datetime
import csv
import functools
import json
import numpy as np
import sys
import ast

def main(sc):
    '''
    Transfer our code from the notebook here, however, remember to replace
    the file paths with the ones provided in the problem description.
    '''
    rddPlaces = sc.textFile('/data/share/bdm/core-places-nyc.csv')
    rddPattern = sc.textFile('/data/share/bdm/weekly-patterns-nyc-2019-2020/*')
    OUTPUT_PREFIX = sys.argv[1]
    CAT_CODES = {'445210', '445110', '722410', '452311', '722513',
             '445120', '446110', '445299', '722515', '311811',
             '722511', '445230', '446191', '445291', '445220',
             '452210', '445292'}
    CAT_GROUP = {'452210': 0, '452311': 0,
             '445120': 1,
             '722410': 2,
             '722511': 3,
             '722513': 4,
             '446110': 5, '446191': 5,
             '722515': 6, '311811': 6,
             '445210': 7, '445299': 7, '445230': 7, '445291': 7, '445220': 7, '445292': 7,
             '445110': 8}
    def filterPOIs(_, lines):
        reader = csv.reader(lines)
        for line in reader:
            group = CAT_GROUP.get(line[9],-1)
            if group >=0:
                (placekey) = (line[0])
                yield (placekey,group)

    rddD = rddPlaces.mapPartitionsWithIndex(filterPOIs) \
            .cache()
    storeGroup = dict(rddD.collect())

    groupCount = rddD.map(lambda x:(x[1],1)) \
        .reduceByKey(lambda x,y: x+y) \
        .sortByKey() \
        .values() \
        .collect()
    def extractVisits(storeGroup, _, lines):
        startDate = datetime.datetime(2019,1,1)
        for row in csv.reader(lines):
            group = storeGroup.get(row[0],-1)
            if group>=0 and int(row[14])>0:
                (placekey, year,date, visits) = (row[0], row[12],row[12], row[16]) 
                date = date.split('T')[0]
                visits = ast.literal_eval(visits)
                if date != None:
                    date_1 = datetime.datetime.strptime(date, "%Y-%m-%d")
                    for i in range (0,7):
                        end_date = date_1 +datetime.timedelta(days=i)
                        day =(end_date - startDate).days
                        end_date_str = str(end_date)
                        visits_by_day = visits[i]
                        end_date_str = end_date_str.split()[0]
                        year = end_date_str.split('-')[0] 
                        month = int(end_date_str.split('-')[1]) 
                        date = int(end_date_str.split('-')[2]) 
                        if year =="2019" or year =="2020":
                            yield((group,day),visits_by_day)
    rddG = rddPattern \
        .mapPartitionsWithIndex(functools.partial(extractVisits, storeGroup))
    def computeStats(groupCount, _, records):
        for record in records:
            (group,days)=record[0]
            visits= list(record[1])
            median = np.median(visits)
            low = (max(0,median - np.std(visits)))
            high = (max(0,median + np.std(visits)))
            yield ((group,days),int(median), low,high)

    rddH = rddG.groupByKey() \
            .mapPartitionsWithIndex(functools.partial(computeStats, groupCount))
    def computeStats(groupCount, _, records):
        startDate = datetime.datetime(2019,1,1)
        for record in records:
            (group,days)=record[0]
            visits= list(record[1])
            median = np.median(visits)
            date = startDate + datetime.timedelta(days=days)
            low = int(max(0,median - np.std(visits)))
            high = int(max(0,median + np.std(visits)))
            year = date.year
            month = date.strftime('%m')
            date_1 = date.strftime('%d')
            year_date = f'2020-{month}-{date_1}'
            yield (group,f'{year},{year_date},{int(median)},{low},{high}')


    rddI = rddG.groupByKey() \
            .mapPartitionsWithIndex(functools.partial(computeStats, groupCount))
    rddJ = rddI.sortBy(lambda x: x[1][:15])
    header = sc.parallelize([(-1, 'year,date,median,low,high')]).coalesce(1)
    rddJ = (header + rddJ).coalesce(10).cache()
    filename = ['big_box_grocers',
            'convenience_stores',
            'drinking_places',
            'full_service_restaurants',
            'limited_service_restaurants',
            'pharmacies_and_drug_stores',
            'snack_and_bakeries',
            'specialty_food_stores',
            'supermarkets_except_convenience_stores']
            
    for i in range (9):
        rddJ.filter(lambda x: x[0]==i or x[0]==-1).values() \
            .saveAsTextFile(f'{OUTPUT_PREFIX}/{filename[i]}')
if __name__=='__main__':
    sc = SparkContext()
    main(sc)
