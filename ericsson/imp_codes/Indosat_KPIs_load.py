import pandas as pd
import numpy as np
import os
import datetime
import sqlalchemy
import UFO
import imaplib
from sqlalchemy import create_engine, MetaData, Table, select
from six.moves import urllib
from email.parser import Parser
from dateutil.parser import parse
p = Parser()

def MAPA(path, file):
    df = pd.read_csv(path + file, sep = ';')

    for col in df.columns:
        if col == list(df[col].unique())[0]:
            df.rename(columns = {col: 'Date'}, inplace=True)
        elif 'Unnamed' in col:
            df.drop(columns = [col], inplace = True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date

    for char in col_chars:
        df.rename(columns = {col: col.replace(char, col_chars[char]) for col in df.columns}, inplace=True)

    df.rename(columns = {'Short name': 'Short_name'}, inplace = True)
    cols_to_proc = [col for col in df.columns if col not in ['Short_name', 'Date']]


    for col in cols_to_proc:

        if len(df.loc[df[col].astype(str).str.contains('%')]) > 0:
            df[col] = df[col].str.replace('%', '').str.replace(' ', '').astype(float)
            if col[-4:] != 'PERC':
                new_name = col + '_PERC'
                df.rename(columns = {col: new_name}, inplace = True)
                cols_to_proc[cols_to_proc.index(col)] = new_name

        elif len(df.loc[df[col].astype(str).str.upper().str.contains('MBPS')]) > 0:
            df[col] = df[col].str.upper().str.replace('MBPS', '').str.replace(' ', '').astype(float)
            if 'MBPS' not in col.upper():
                new_name = col + '_MBPS'
                df.rename(columns = {col: new_name}, inplace = True)
                cols_to_proc[cols_to_proc.index(col)] = new_name

        elif len(df.loc[df[col].astype(str).str.upper().str.contains('TB')]) > 0:
            df[col] = df[col].str.upper().str.replace('TB', '').str.replace(' ', '').astype(float)
            if col[-2:] != 'TB':
                new_name = col + '_TB'
                df.rename(columns = {col: new_name}, inplace = True)
                cols_to_proc[cols_to_proc.index(col)] = new_name

        elif len(df.loc[df[col].astype(str).str.upper().str.contains('MS')]) > 0:
            df[col] = df[col].str.upper().str.replace('MS', '').str.replace(' ', '').astype(float)
            if col[-2:] != 'MS':
                new_name = col + '_MS'
                df.rename(columns = {col: new_name}, inplace = True)
                cols_to_proc[cols_to_proc.index(col)] = new_name

        elif len(df.loc[df[col].astype(str).str.upper().str.slice(start=-1) == 'E']) > 0:
            df[col] = df[col].str.upper().str.replace('E', '').str.replace(' ', '').astype(float)
            if col[-1:] != 'E':
                new_name = col + '_E'
                df.rename(columns = {col: new_name}, inplace = True)
                cols_to_proc[cols_to_proc.index(col)] = new_name


    df['LOADTIMESTAMP'] = datetime.datetime.now()
    df['LOADTIMESTAMP'] = pd.to_datetime(pd.to_datetime(df['LOADTIMESTAMP']).dt.strftime("%Y-%m-%d %H:%M:%S"))    
    df.rename(columns = {col: col.upper() for col in df.columns}, inplace = True)
    
    return df

def IPLB(path, file):
    df = pd.read_csv(path + file)
    df.rename(columns = {col: col.replace(' ', '_').replace('%', 'PERC').upper() for col in df.columns}, inplace = True)
    for col in df.columns:
        if 'TIME' in col:
            df[col] = pd.to_datetime(df[col]).dt.date
            
    df['LOADTIMESTAMP'] = datetime.datetime.now()
    df['LOADTIMESTAMP'] = pd.to_datetime(pd.to_datetime(df['LOADTIMESTAMP']).dt.strftime("%Y-%m-%d %H:%M:%S"))
            
    return df

def BH(path, file):
    df = pd.read_csv(path + file)
    df['period_start_time'] = pd.to_datetime(df['period_start_time'] + ' ' + df.columns[0])
    df.drop(columns = [df.columns[0]], inplace = True)

    df.rename(columns = {col: col.upper() for col in df.columns}, inplace = True)
    df.rename(columns = {col: col.replace('(TAKE 20 HOUR VALUE)', '') for col in df.columns}, inplace = True)
    df.rename(columns = {col: col.replace('BUSY HOUR', '').replace('-', '')
                                                          .replace('_', '')
                                                          .replace('/', '')
                                                          .replace('(', '')
                                                          .replace(')', '')
                                                          .replace(' ', '')
                                                           for col in df.columns}, inplace = True)
    df.rename(columns = {'PERIODSTARTTIME': 'PERIOD_START_TIME',
                         'TOTALTRAFFICERLERLANG': 'TOTAL_TRAFFIC_ERLANG',
                         'TOTALCALLATTEMPTS': 'TOTAL_CALL_ATTEMPTS',
                         'TOTALATTACHEDVLRSUBSCRIBER': 'TOTAL_ATTACHED_VLR_SUBSCRIBER',
                         'TRAFFICSUB': 'TRAFFIC_SUB',
                         'BHCASUB': 'BHCA_SUB',
                         'TOTALDETACH': 'TOTAL_DETACH',
                         'TOTALATTACH': 'TOTAL_ATTACH',
                         'TOTALVLRREGISTER': 'TOTAL_VLR_REGISTER'}, inplace = True)
    for col in df.columns:
        if 'TIME' in col:
            df[col] = pd.to_datetime(df[col])
            
    df['LOADTIMESTAMP'] = datetime.datetime.now()
    df['LOADTIMESTAMP'] = pd.to_datetime(pd.to_datetime(df['LOADTIMESTAMP']).dt.strftime("%Y-%m-%d %H:%M:%S"))

    return df

def KPI(path, file):
    df = pd.read_csv(path + file)
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]].apply(parse)).dt.date
    df.rename(columns = {df.columns[0]: 'DATE'}, inplace = True)
    df.rename(columns = {col: col.replace(' / ', '_')
                                 .replace('%', 'perc')
                                 .replace(';', '_')
                                 .replace('&', '_')
                                 .replace(' ', '_')
                                 .replace('-', '_')
                                 .upper() for col in df.columns}, inplace = True)

    df['LOADTIMESTAMP'] = datetime.datetime.now()
    df['LOADTIMESTAMP'] = pd.to_datetime(pd.to_datetime(df['LOADTIMESTAMP']).dt.strftime("%Y-%m-%d %H:%M:%S"))
    
    return df

col_chars = {'>=': '_GE_',
             '<=': '_LE_',
             '>': '_GT_',
             '<': '_LT_',
             '%': 'PERC',
             '-': '_',
             '.': '_'}

subjects = ['mapa_report',
            'Indosat_Tableau_Core_Daily_KPI_IPLB',
            'Indosat_Tableau_Core_Daily_KPI_BH',
            'Indosat_Tableau_Core_KPI',
            'Indosat_Tableau_daily_data_Core_CPU_Load_Exclusion'] # <---- add subject here


path = 'E:\\00.SoftwareDev\\Projects\\Indosat_KPIS\\Data\\'
already_used = list(set(pd.read_csv(path.replace('Data\\', '') + 'used_files_MAPA.csv', sep = ',')['used']))
today = str(datetime.date.today().strftime("%d-%b-%Y"))
yesterday = str((datetime.date.today() - datetime.timedelta(days = 1)).strftime("%d-%b-%Y"))
emails = []
to_upload = []

params = urllib.parse.quote_plus("""DRIVER={};SERVER=ESESSMW4310.ss.sw.ericsson.se\MSSQL_TBL_DEV;
DATABASE=FE_Indosat-Indonesia;UID={};PWD={}""".format('{SQL Server Native Client 11.0}',
                                            UFO.u4310, UFO.p4310))
engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params, fast_executemany=True)
engine.connect()

conn = imaplib.IMAP4_SSL(UFO.ia,993)
conn.login(UFO.iu,UFO.ip)
conn.select("INBOX")


for subject in subjects:
    _msg_, indexes = conn.search(None, 'Subject "{}" SENTSINCE {}'.format(subject, yesterday))
    emails = emails + [x.decode("utf-8") for x in indexes[0].split()]
    
for index in emails:
    email = p.parsestr(conn.fetch(index,"(RFC822)")[1][0][1].decode("utf-8"))
    email_date = parse(email.get('Date')).date().strftime("%d-%b-%Y")
    if email_date == today:
        for i, part in enumerate(email.walk()):
            if part.get_content_maintype() == 'multipart':
                continue
            filename = part.get_filename()
            if filename and '.csv' in filename.lower():
                new_filename = filename.replace('.csv', '_' + email_date + '.csv').replace('-', '_')
                if new_filename not in already_used:
                    to_upload.append(new_filename)
                    with open(path + new_filename, 'wb') as fp:
                        fp.write(part.get_payload(decode=True))

files = [file for file in os.listdir(path)]
already_used = list(set(to_upload + already_used))
log = pd.DataFrame(data = {'used': already_used})
log.to_csv(path.replace('Data\\', '') + 'used_files_MAPA.csv', index=False, sep = ',')

for file in to_upload:
    if 'Daily_MAPA_report_Tableau' in file:
        MAPA(path, file).to_sql(name='MAPA_Report', schema = 'dbo', con=engine, index=False, if_exists='append')
        
    elif 'Tableau_Core_KPI_IPLB' in file:
        IPLB(path, file).to_sql(name='Core_KPI_IPLB', schema = 'dbo', con=engine, index=False, if_exists='append')
        
    elif 'Tableau_Core_Daily_BH_KPI' in file:
        BH(path, file).to_sql(name='Core_BH_KPI', schema = 'dbo', con=engine, index=False, if_exists='append')
        
    elif 'Tableau_Core_Daily_KPI' in file:
        KPI(path, file).to_sql(name='Core_Daily_KPI', schema = 'dbo', con=engine, index=False, if_exists='append')
        
    elif 'Tableau_Core_Daily_CPU_Load_Exclusion' in file:
        CPU(path, file).to_sql(name='Core_CPU_Load_Exclusion', schema = 'dbo', con=engine, index=False, if_exists='append')

conn.close()
engine.dispose()

if (datetime.datetime.now().day == 1) and (datetime.datetime.now().month % 2 == 0):
    to_del = os.listdir(path)
    for file in to_del:
        os.remove(path + file)