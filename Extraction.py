# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:07:24 2020

@author: mdevasish
"""
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table


if __name__ == '__main__':
    engine = create_engine('sqlite:///data/traffic.db')
    metadata = MetaData()

    traffic = Table('traffic', metadata, autoload=True, autoload_with=engine)

    query = "select * from traffic where date_time >= '2013-01-01'"
    results = engine.execute(query).fetchall()

    x = pd.DataFrame(results,columns=traffic.columns.keys())
    x.to_csv('./data/traffic.csv',index = False)
    