# Libraries
import pandas as pd
from sqlalchemy import create_engine
import urllib
import pyodbc


def insert_data_to_sql(data, table_name, schema_name):
    server = 'BRARJ2DBSQL01'
    database = 'DIGITAL_PH_DEV'
    username = 'BRAXCARPH1'
    password = 'yKla2dJaG4wmVmK3zrJ$'
    quoted = urllib.parse.quote_plus(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=BRARJ2DBSQL01;DATABASE=DIGITAL_PH_DEV;UID=' + username + ';PWD=' + password)
    engine = create_engine('mssql+pyodbc:///?odbc_connect={}'.format(quoted))
    data.to_sql(table_name, schema=schema_name, con=engine, if_exists='replace', index=False)
    print('Upload of {} to {} schema: {} completed'.format(table_name, database, schema_name))


table_n = 'last_month_clientlist'

# CHILE
cl_old = pd.read_excel(
    r'C:\Users\b4cy01\OneDrive - Linde Group\001 - Customer At Risk\Chile\PowerApp_Files\car_cl_clientes.xlsx')
schema_n = 'car_cl'
try:
    insert_data_to_sql(cl_old, table_n, schema_n)
except:
    print('Upload of {} to {} schema: {} FAILED'.format(table_n, 'DIGITAL_PH_DEV', schema_n))

# ARGENTINA
ar_old = pd.read_excel(
    r'C:\Users\b4cy01\OneDrive - Linde Group\001 - Customer At Risk\Argentina\PowerApp_Files\car_ar_clientes.xlsx')
schema_n = 'car_ar'
try:
    insert_data_to_sql(ar_old, table_n, schema_n)
except:
    print('Upload of {} to {} schema: {} FAILED'.format(table_n, 'DIGITAL_PH_DEV', schema_n))

# PERU
pe_old = pd.read_excel(
    r'C:\Users\b4cy01\OneDrive - Linde Group\001 - Customer At Risk\Peru\PowerApp_Files\car_pe_clientes.xlsx')
schema_n = 'car_pe'
try:
    insert_data_to_sql(pe_old, table_n, schema_n)
except:
    print('Upload of {} to {} schema: {} FAILED'.format(table_n, 'DIGITAL_PH_DEV', schema_n))

# BOLIVIA
bo_old = pd.read_excel(
    r'C:\Users\b4cy01\OneDrive - Linde Group\001 - Customer At Risk\Bolivia\PowerApp_Files\car_bo_clientes.xlsx')
schema_n = 'car_bo'
try:
    insert_data_to_sql(bo_old, table_n, schema_n)
except:
    print('Upload of {} to {} schema: {} FAILED'.format(table_n, 'DIGITAL_PH_DEV', schema_n))

# PARAGUAY
py_old = pd.read_excel(
    r'C:\Users\b4cy01\OneDrive - Linde Group\001 - Customer At Risk\Paraguay\PowerApp_Files\car_py_clientes.xlsx')
schema_n = 'car_py'
try:
    insert_data_to_sql(py_old, table_n, schema_n)
except:
    print('Upload of {} to {} schema: {} FAILED'.format(table_n, 'DIGITAL_PH_DEV', schema_n))

# URUGUAY
uy_old = pd.read_excel(
    r'C:\Users\b4cy01\OneDrive - Linde Group\001 - Customer At Risk\Uruguay\PowerApp_Files\car_uy_clientes.xlsx')
schema_n = 'car_uy'
try:
    insert_data_to_sql(uy_old, table_n, schema_n)
except:
    print('Upload of {} to {} schema: {} FAILED'.format(table_n, 'DIGITAL_PH_DEV', schema_n))