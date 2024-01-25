import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import date
import pyodbc

"""
UnusedPackeges:

import datetime
import time
import pickle

Specific lines for Jupyter Notebook:

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.2f}'.format
% autosave
120
"""



# Export into .xlsx files for the app.
def export_to_excel(filename, table_name, df):
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', startrow=1, header=False, index=False)
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    (max_row, max_col) = df.shape
    column_settings = []
    for header in df.columns:
        column_settings.append({'header': header})
    worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings, "name": table_name})
    worksheet.set_column(0, max_col - 1, 12)
    writer.save()


def categorize_risk_pe(df_client_i, df_risk_i, c_col):
    df_client = df_client_i.copy()
    df_risk = df_risk_i.copy()
    df_risk['Risk_Level_n'] = np.where(df_risk['Risk_Level'].isin(['Perdido', 'En Riesgo Alto', 'En Riesgo']), 2,
                                       np.where(df_risk['Risk_Level'] == 'Oportunidad', 1, 0))
    df_m = df_risk[[c_col, 'Risk_Level_n']].groupby([c_col]).max().reset_index()
    df_out = df_client.merge(df_m, how='left', on=[c_col])
    df_out['Risk_Level_n'] = np.where(df_out['Risk_Level_n'] == 2, 'Risk',
                                      np.where(df_out['Risk_Level_n'] == 1, 'Opportunity', 'Out'))
    return df_out


def categorize_risk(df_client_i, df_risk_i, c_col, s_col):
    df_client = df_client_i.copy()
    df_risk = df_risk_i.copy()
    df_risk['Risk_Level_n'] = np.where(df_risk['Risk_Level'].isin(['Perdido', 'En Riesgo Alto', 'En Riesgo']), 2,
                                       np.where(df_risk['Risk_Level'] == 'Oportunidad', 1, 0))
    df_m = df_risk[[c_col, s_col, 'Risk_Level_n']].groupby([c_col, s_col]).max().reset_index()
    df_client[c_col] = df_client[c_col].astype(str)
    df_m[c_col] = df_m[c_col].astype(str)
    df_client[s_col] = df_client[s_col].astype(str)
    df_m[s_col] = df_m[s_col].astype(str)
    df_out = df_client.merge(df_m, how='left', on=[c_col, s_col])
    df_out['Risk_Level_n'] = np.where(df_out['Risk_Level_n'] == 2, 'Risk',
                                      np.where(df_out['Risk_Level_n'] == 1, 'Opportunity', 'Out'))
    return df_out


# Diccionario de Pais: (CLientes,ProductosRiesgo,RutaExport)
tables_dict = {'cl': 'Chile',
               'uy': 'Uruguay',
               'ar': 'Argentina',
               'pe': 'Peru',
               'py': 'Paraguay',
               'bo': 'Bolivia'}

# Parámetros de conexión
server = 'BRARJ2DBSQL01'
database = 'DIGITAL_PH_DEV'
username = 'BRAXCARPH1'
password = 'yKla2dJaG4wmVmK3zrJ$'
cnxn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
cursor = cnxn.cursor()

# Query básica
query = 'SELECT * FROM car_{}.{} WHERE periodo_analisis = {}'
query2 = 'SELECT * FROM car_{}.{} WHERE periodo = {}'
query_pe = 'SELECT * FROM car_pe.pe_car_contactdata'
var_periodo = (date.today() + relativedelta(months=-1)).strftime("%Y%m")

# Loop
for key in tables_dict:
    print('---------------------')
    print(f'Processing {key}')

    if key not in ['cl']:
        data_clientes = pd.read_sql(query.format(key, 'lista_clientes', var_periodo), cnxn)
    else:
        data_clientes = pd.read_sql(query2.format(key, 'lista_clientes', var_periodo), cnxn)
    data_productos = pd.read_sql(query.format(key, 'prod_riesgo', var_periodo), cnxn)

    if key == 'cl':
        data_clientes_app = categorize_risk(data_clientes, data_productos, 'cliente', 'sucursal')
    elif key == 'pe':
        data_clientes_app = categorize_risk_pe(data_clientes, data_productos, 'CodigoClienteFull')
    elif key == 'ar':
        data_clientes_app = categorize_risk(data_clientes, data_productos, 'CODIGO', 'AGENCIA')
    elif key == 'bo':
        data_clientes_app = categorize_risk(data_clientes, data_productos, 'NO_CLIENTE', 'CENTROD')
    elif key == 'py':
        data_clientes_app = categorize_risk(data_clientes, data_productos, 'NO_CLIENTE', 'CENTRO')
    elif key == 'uy':
        data_clientes_app = categorize_risk(data_clientes, data_productos, 'cliente', 'sucursal')

    export_to_excel(
        r'C:\Users\ar12bc\OneDrive - Linde Group\001 - Customer At Risk\Prueba\{}\PowerApp_Files\car_{}_clientes.xlsx'.format(
            tables_dict[key], key),
        str.upper(key) + '_CAR_clients', data_clientes_app)
    export_to_excel(
        r'C:\Users\ar12bc\OneDrive - Linde Group\001 - Customer At Risk\Prueba\{}\PowerApp_Files\car_{}_reg.xlsx'.format(
            tables_dict[key], key),
        str.upper(key) + '_CAR_data', data_productos)

    print(f'tamanho da base clientes para {tables_dict[key]} : {data_clientes_app.shape}')
    print(f'tamanho da base produtos para {tables_dict[key]} : {data_productos.shape}')
    print(f'{tables_dict[key]} done')

    if key == 'pe':
        data_contact = pd.read_sql(query_pe, cnxn)
        export_to_excel(
            r'C:\Users\ar12bc\OneDrive - Linde Group\001 - Customer At Risk\Prueba\{}\PowerApp_Files\car_{}_contacto.xlsx'.format(
                tables_dict[key], key),
            str.upper(key) + '_CAR_contacto', data_productos)

    print('---------------------')
