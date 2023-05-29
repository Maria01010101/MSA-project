import streamlit as st
from typing import Union
import bs4
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import urllib
from urllib.request import urlretrieve
import time
import csv
import numpy as np
import random

def get_ua():
    user_agents = [
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36 OPR/26.0.1656.60',
        'Opera/8.0 (Windows NT 5.1; U; en)',
        'Mozilla/5.0 (Windows NT 5.1; U; en; rv:1.8.1) Gecko/20061208 Firefox/2.0.0 Opera 9.50',
        'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; en) Opera 9.50',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0',
        'Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.57.2 (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2 ',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
        'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 Safari/534.16',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.11 TaoBrowser/2.0 Safari/536.11',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER',
        'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)',
        'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.84 Safari/535.11 SE 2.X MetaSr 1.0',
        'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SV1; QQDownload 732; .NET4.0C; .NET4.0E; SE 2.X MetaSr 1.0) ',
    ]
    user_agent = random.choice(user_agents) # 随机抽取对象
    return user_agent

url1='https://datacenter-web.eastmoney.com/api/data/v1/get'
ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
headers={'User-Agent':ua}
params={
    'sortColumns': 'SECURITY_CODE',
    'sortTypes': '1',
    'pageSize': '5000',
    'pageNumber': '1',
    'reportName':'RPT_VALUEANALYSIS_DET',
    'columns': 'ALL',
    'quoteColumns':'' ,
    'source': 'WEB',
    'client': 'WEB',
    'filter': f'''(TRADE_DATE='{time.strftime("%Y-%m-%d",time.localtime(time.time()-60*60*24))}')'''
}
res=requests.get(url=url1,headers=headers,params=params).json()

count=0

df = pd.DataFrame()
for each in res['result']['data']:
    count+=1
    # print(each['SECUCODE'])
    secid='0'+'.'+each['SECUCODE'][:6] if each['SECUCODE'][-2:] == 'SZ' else '1'+'.'+each['SECUCODE'][:6]
    url_each="http://push2.eastmoney.com/api/qt/stock/get"
    params_each = {
        'fitt':'1',
        'invt':'2',
        'fields':'f57,f107,f162,f152,f167,f92,f59,f183,f184,f105,f185,f186,f187,f173,f188,f84,f116,f85,f117,f190,f189,f62,f55',
        'secid':secid
    }
    res_each = requests.get(url=url_each,headers=headers,params=params_each).json()
    dic={
        '序号':res['result']['data'].index(each)+1,
        '股票代码':each['SECURITY_CODE'],
        '股票简称':each['SECURITY_NAME_ABBR'], 
        '最新价':each['CLOSE_PRICE'],
        '涨跌幅(%)':each['CHANGE_RATE'],
        'PE(TTM)':each['PE_TTM'],
        'PE(静)':each['PE_LAR'],
        '市净率':each['PB_MRQ'],
        'PEG值':each['PEG_CAR'],
        '市销率':each['PS_TTM'],
        '市现率':each['PCF_OCF_TTM'],
        '所属行业':each['BOARD_NAME'],
        '收益': res_each['data']['f55'],
        '总股本':res_each['data']['f84'],
        'ROE(%)':res_each['data']['f173'],
        '总股本':res_each['data']['f84'],
        '净利率(%)':res_each['data']['f187'],
        '负债率(%)':res_each['data']['f188'],
        '同比(%)':res_each['data']['f185'],
        '每股净资产':res_each['data']['f92'],
        '毛利率':res_each['data']['f186'],
        'PE(动)':res_each['data']['f162'],
        '净利润':res_each['data']['f105'],
        '总营收':res_each['data']['f183'],
        '总值':res_each['data']['f116'],
        '流值':res_each['data']['f117']
    }
    # print(count)
    # print(dic)
    column_names = list(dic.keys())  # 列名列表
    df_each=pd.DataFrame(np.array(list(dic.values())).reshape(1,-1),columns=column_names)
    df=pd.concat([df,df_each])
    time.sleep(2)
    st.dataframe(df)
