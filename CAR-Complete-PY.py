# Libraries
import pandas as pd, numpy as np
from dateutil.relativedelta import relativedelta
import datetime
from datetime import date
import time
from sqlalchemy import create_engine
import urllib
import pyodbc
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn import metrics

"""
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.2f}'.format
%autosave 120
"""

def get_sql_data(query):
    server = 'BRARJ2DBSQL01' 
    database = 'DIGITAL_PH_DEV' 
    username = 'BRAXCARPH1' 
    password = 'yKla2dJaG4wmVmK3zrJ$'
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()
    data = pd.read_sql(query,cnxn)
    return data

def process_sales(data_input):
    data_1 = data_input.copy()   
    data_1['periodo'] = pd.to_datetime(data_1['PERIODO'], format = '%m/%Y').dt.strftime('%Y%m')
    data_1 = data_1.rename(columns={'SUBSTR(VM.TIPO_CLIENTE,1,2)':'TIPO_CLIENTE',
                                    'SUBSTR(VM.TIPO_CLIENTE,4,15)':'TIPO_CLIENTE_2'})
    data_1['TIPO_DATOS'] = np.select(
        [
            data_1['TIPO_DATOS'] == 'CO2 LÃ\xadquido',
            data_1['TIPO_DATOS'] == 'MercaderÃ\xadas',
            data_1['TIPO_DATOS'] == 'NitrÃ³geno Gaseoso',
            data_1['TIPO_DATOS'] == 'ArgÃ³n',
            data_1['TIPO_DATOS'] == 'NitrÃ³geno LÃ\xadquido'
        ], 
        [
            'CO2 Liquido',
            'Mercaderias',
            'Nitrogeno Gaseoso',
            'Argon',
            'Nitrogeno Liquido'
        ], 
        default=data_1['TIPO_DATOS']
    )
    print(len(data_1))
    return data_1

# Calculate total sales last 12M (including services)
def calculate_total_sale_last_12M(df_input):
    df_analysis = df_input.copy()
    time_max = pd.to_datetime(df_analysis['periodo'].max(), format='%Y%m')
    time_min = time_max + relativedelta(months=-11)
    df_analysis = df_analysis[(df_analysis['FECHA_FACTURA'] >= time_min)]
    df_analysis = df_analysis.groupby(['NO_CLIENTE','CENTRO']).agg({'VTA_RECONOCIDA':'sum'}).reset_index()
    return df_analysis

# Reduce data to the id-period-product level. (Group all invoices/creditnotes/debitnotes inside a singular month)
def data_step2a_process_idperiod(df_input, id_column, product_column, time_column, suc_column,
                                 sales_column, mark_column, volume_column):
    
    df = df_input.copy()

    df_mv = df.groupby([id_column,time_column, product_column, suc_column]).agg({sales_column: 'sum'}).reset_index()
    df_mv.columns.values[0], df_mv.columns.values[1] = id_column,time_column
    df_mv.columns.values[2], df_mv.columns.values[3] = product_column, suc_column
    df_mv.columns.values[4] = 'monetary_value'

    df_freq = df.groupby([id_column,time_column,product_column, suc_column]).agg({mark_column: 'count'}).reset_index()
    df_freq.columns.values[0], df_freq.columns.values[1] = id_column,time_column
    df_freq.columns.values[2], df_freq.columns.values[3] = product_column, suc_column
    df_freq.columns.values[4] = 'frequency'

    df_others = df.groupby([id_column,time_column,product_column, suc_column]).agg({volume_column:'sum'}).reset_index()
    df_others.columns.values[0], df_others.columns.values[1] = id_column,time_column
    df_others.columns.values[2], df_others.columns.values[3] = product_column, suc_column
    df_others.columns.values[4] = 'volume'

    df2 = df_mv.merge(df_freq,how='left',on=[id_column,time_column, product_column, suc_column])
    df2 = df2.merge(df_others,how='left',on=[id_column,time_column, product_column, suc_column])
    df2['frequency'] = np.where(df2['frequency'] >= 1, 1, 0)
    
    df_maxtbp = df2.sort_values(by=[id_column,time_column, product_column, suc_column])
    df_maxtbp=df_maxtbp.merge(df_maxtbp.groupby([id_column,product_column])[time_column].shift().rename('prev_periodo'),
                                how='left', left_index=True, right_index=True)
    df_maxtbp['prev_periodo'] = np.where(df_maxtbp['prev_periodo'].isna(), df_maxtbp[time_column], df_maxtbp['prev_periodo'])
    df_maxtbp['base_periodo'] = pd.to_datetime(df_maxtbp[time_column], format='%Y%m')
    df_maxtbp['prev_periodo'] = pd.to_datetime(df_maxtbp['prev_periodo'], format='%Y%m')
    df_maxtbp['max_time_between_tr'] = (df_maxtbp['base_periodo'].dt.year - df_maxtbp['prev_periodo'].dt.year)*12+ \
                                 ((df_maxtbp['base_periodo'].dt.month - df_maxtbp['prev_periodo'].dt.month))
    df_maxtbp = df_maxtbp[[id_column,time_column, product_column, suc_column,'max_time_between_tr']]
    
    df2 = df2.merge(df_maxtbp, how='left', on=[id_column,time_column, product_column, suc_column])
    
    return df2

# Get last transaction on the last 13-24 months. (for average time between purchases feature)
def data_step2b_last_transaction(df_input, training_window, analysis_period, analysis_window, col_groupby, op):
    df = df_input.copy()
    analysis_end = pd.to_datetime(analysis_period, format='%Y%m') + relativedelta(months=-training_window)
    analysis_start = analysis_end + relativedelta(months=-analysis_window)
    df = df[(pd.to_datetime(df['periodo'], format='%Y%m') >= analysis_start) &
            (pd.to_datetime(df['periodo'], format='%Y%m') <= analysis_end)]
    
    df = df.groupby(col_groupby).agg({'periodo':op}).rename(columns={'periodo':'Purchase_Ref_Old'})
    return df

# Compress data of all periods into one single row. id-product level
def data_step3(df_input, config, analysis_period_input, attribute_window, id_column, time_column, prod_column, suc_column,
               df_past):

    df = df_input.copy()
    
    analysis_period = pd.to_datetime(analysis_period_input, format='%Y%m')
    analysis_period_dr = analysis_period_input
    
    window_max = pd.to_datetime(analysis_period, format='%Y%m')
    window_start = pd.to_datetime(analysis_period, format='%Y%m') + relativedelta(months=-attribute_window)

    df_att=df[(pd.to_datetime(df[time_column], format='%Y%m') <= window_max) &
             (pd.to_datetime(df[time_column], format='%Y%m') > window_start)]

    fix_period = pd.to_datetime(df_att['periodo'].min(), format='%Y%m')
    df_final = df_att.groupby([id_column,suc_column,prod_column]).agg(config).reset_index()
    df_final.loc[:, 'recency'] = fix_period - pd.to_datetime(df_final['periodo'], format='%Y%m')
    df_final.loc[:, 'recency'] = df_final['recency'].apply(lambda x: 0 if pd.isnull(x) else round(abs(x.days)/30.4375,0)+1)
    df_final.rename(columns={'periodo':'last_purchase'}, inplace=True)
    
    df_past_t2 = df_past.merge(df_final[[id_column,suc_column,prod_column,
                                         'last_purchase','frequency']], on=[id_column,suc_column,prod_column])
                                                                                                         
    lp = pd.to_datetime(df_past_t2['last_purchase'], format='%Y%m')
    op = pd.to_datetime(df_past_t2['Purchase_Ref_Old'], format='%Y%m')
    df_past_t2['Mean_time_between_tr'] = np.where(df_past_t2['Ref_Old'] == 'recent', 
                    (((lp.dt.year - op.dt.year)*12 + (lp.dt.month - op.dt.month)+1)/df_past_t2['frequency']),
                    (((lp.dt.year - op.dt.year)*12 + (lp.dt.month - op.dt.month)+1)/(df_past_t2['frequency']+1)))

    
    df_past_t2 = df_past_t2.drop(columns=['last_purchase','Ref_Old','Purchase_Ref_Old', 'frequency'])    
    df_final = df_final.merge(df_past, how='left', on=[id_column,suc_column,prod_column])
    df_final = df_final.merge(df_past_t2, how='left', on=[id_column,suc_column,prod_column])     
    df_final['periodo_analisis'] = analysis_period_dr
    df_final = df_final.fillna(0)
      
    return df_final

def data_step3_target_for_cv(df_input, analysis_period, performance_window, id_column, time_column, prod_column):
    df = df_input.copy()
    performance_ends = pd.to_datetime(analysis_period, format='%Y%m')
    performance_begins = pd.to_datetime(analysis_period, format='%Y%m') + relativedelta(months=-performance_window)
    df_perf = df[(pd.to_datetime(df[time_column], format='%Y%m') > performance_begins) &
              (pd.to_datetime(df[time_column], format='%Y%m') <= performance_ends)] 
    df_perf = df_perf.groupby([id_column,prod_column]).agg({'monetary_value':['sum','mean']})
    df_perf.columns = df_perf.columns.droplevel(0)
    df_perf = df_perf.rename(columns={'sum':'future_spend_total','mean':'future_spend_avg'})
    df_perf['future_spend_avg'] = df_perf['future_spend_avg'].replace(np.nan, 0)
    df_perf['future_spend_total'] = df_perf['future_spend_total'].replace(np.nan, 0)
    df_perf['target'] = np.where(df_perf['future_spend_total'].isna(), 1, 0)
    df_perf = df_perf.reset_index()
    return df_perf

# Perform cap/flooring process
def data_step4a(df_input, capping, cap, flooring, floor, proc_list):
    df = df_input.copy()
    if capping:
            q = df.quantile(cap)
            for var in q.index:
                if var in proc_list:
                    df[var] = np.minimum(df[var], q.loc[var])

    if flooring:
            f = df.quantile(floor)
            for var in f.index:
                if var in proc_list:
                    df[var] = np.maximum(df[var], f.loc[var])
                
    return df

# One hot encode categorical values for model
def data_step4b(df_input, columns_to_encode):
    df = df_input.copy()
    encoded = pd.get_dummies(df[columns_to_encode])
    encoded = encoded.drop(columns=['CENTRO','TIPO_CLIENTE'])
    return df.merge(encoded, how='left', left_index=True,right_index=True)

def calibrate_model(df_input, save = False):
    
    # Filter data
    data_pre = df_input.copy()
    print('Original data length:', len(data_pre))
    data_pre=data_pre[data_pre['frequency']>3] 
    print('Filter by frequent customers:', len(data_pre))
    print('Original Loyal:{:.2%}'.format(len(data_pre[data_pre['target']==0])/len(data_pre)))
    print('Original Churners:{:.2%}'.format(len(data_pre[data_pre['target']==1])/len(data_pre)))
    
    pct=(data_pre.groupby('target').size()/len(data_pre)).iloc[1]/(data_pre.groupby('target').size()/len(data_pre)).iloc[0]
    data_0s = data_pre[data_pre['target']==0].sample(frac=pct, random_state=11)
    data_1 = data_pre[data_pre['target']==1]
    data = pd.concat([data_0s,data_1], ignore_index=True)
    print('Final balanced dataset:', len(data))
            
    x = data.drop(columns=['SECTOR','NO_CLIENTE','CENTRO','TIPO_DATOS','Purchase_Ref_Old','Ref_Old',
                        'periodo_analisis','TIPO_CLIENTE','TIPO_CLIENTE_2','FAMILIA','target'])                            
    y = data['target']
    
    start = time.time()
    n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 2000, num = 6)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 90, num = 5)]
    max_depth.append(None)
    min_samples_split = [2, 4, 6, 8, 10]
    min_samples_leaf = [1, 2, 4, 6]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 4,
                                   verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(x, y)
    print('Time taken to CV: {:.2f} minutes'.format((time.time() - start)/60))
    
    # Save Model
    if save:
        Pkl_Filename = "CAR_CL_Model.pkl"  
        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(rf_random, file)
            
    return rf_random

# Generate metrics for CAR
def generate_metrics(df_input):
    
    print('Calculating model metrics')
    
    # Filter data
    data_pre = df_input.copy()
    data_pre=data_pre[data_pre['frequency']>3] 
    
    pct=(data_pre.groupby('target').size()/len(data_pre)).iloc[1]/(data_pre.groupby('target').size()/len(data_pre)).iloc[0]
    data_0s = data_pre[data_pre['target']==0].sample(frac=pct, random_state=11)
    data_1 = data_pre[data_pre['target']==1]
    data = pd.concat([data_0s,data_1], ignore_index=True)
            
    x = data.drop(columns=['SECTOR','NO_CLIENTE','CENTRO','TIPO_DATOS','Purchase_Ref_Old','Ref_Old',
                        'periodo_analisis','TIPO_CLIENTE','TIPO_CLIENTE_2','FAMILIA','target'])                            
    y = data['target']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)
    
    start = time.time()
    n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 2000, num = 6)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 90, num = 5)]
    max_depth.append(None)
    min_samples_split = [2, 4, 6, 8, 10]
    min_samples_leaf = [1, 2, 4, 6]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 4,
                                   verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(x_train, y_train)
    
    # Metrics
    y_pred_proba = rf_random.predict_proba(x_test)
    y_pred = np.where(pd.DataFrame(y_pred_proba)[1] > 0.50, 1, 0)
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    
    output_metrics = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().reset_index()
    output_metrics = output_metrics.rename(columns={'index':'class'})
    new_row = {'class':'AUC', 'precision':roc_auc, 'recall':roc_auc, 'f1-score':roc_auc, 'support':roc_auc}
    output_metrics = output_metrics.append(new_row, ignore_index=True)
    churn_rate = (len(data_pre[data_pre['target']==1])/len(data_pre))
    new_row2 = {'class':'Churn Rate', 'precision':churn_rate, 'recall':churn_rate, 'f1-score':churn_rate, 'support':churn_rate}
    output_metrics = output_metrics.append(new_row2, ignore_index=True)
    output_metrics['País'] = 'Paraguay'
    output_metrics['Periodo'] = (date.today() + relativedelta(months=-1)).strftime("%Y%m")
    
    
    print('Model metrics - Finished')
            
    return output_metrics

# Model function. Loads the model and run the predictions on production data.
def model_and_predict(df_input, rf_calibrated, load=False):
    
    # Filter data
    data = df_input.copy()
    
    x = data.drop(columns=['SECTOR','NO_CLIENTE','CENTRO','TIPO_DATOS','Purchase_Ref_Old','Ref_Old',
                        'periodo_analisis','TIPO_CLIENTE','TIPO_CLIENTE_2','FAMILIA']) 
    
    # Load Model
    if load:
        Pkl_Filename = r'C:\Users\c1bo91\OneDrive - Linde Group\Proyectos\001 - Customer At Risk\CAR%20-%20PHS\CAR_CL_Model.pkl' 
        with open(Pkl_Filename, 'rb') as file:  
            rf_calibrated = pickle.load(file)        
    # Predict using an existing model on memory
#     y_pred_proba = rf_calibrated.predict_proba(x)
#     y_pred = np.where(pd.DataFrame(y_pred_proba)[1] > 0.50, 1, 0)
    data['Churn_Probability'] = (pd.DataFrame(rf_calibrated.predict_proba(x)))[1]
    return data

def risk_classification_gen(df_input):
    df_result = df_input.copy() 
    max_date = pd.to_datetime(df_result['periodo_analisis'].max(), format='%Y%m') + relativedelta(months=-2)
    conditions  = [(df_result['frequency'] <= 3),
                   (df_result['Churn_Probability'] >= 0.80),
                   (df_result['Churn_Probability'] >= 0.70) & (df_result['Churn_Probability'] < 0.80),
                   (df_result['Churn_Probability'] >= 0.5) & (df_result['Churn_Probability'] < 0.70),
                   (df_result['Churn_Probability'] >= 0.0) & (df_result['Churn_Probability'] < 0.5)
                  ]
    # Lost: (pd.to_datetime(df_result['last_purchase'], format='%Y%m') < max_date)
    risk_levels     = ["Oportunidad","Perdido","En Riesgo Alto","En Riesgo","Estable"]
    df_result['Risk_Level'] = np.select(conditions, risk_levels, default='Unknown') 
    return df_result

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
    worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings,"name": table_name})
    worksheet.set_column(0, max_col - 1, 12)
    writer.save()
    
reference_time = pd.to_datetime('today') + relativedelta(months=-27)
query = '''
SELECT *
  FROM [DIGITAL_PH_DEV].[car_py].[py_car_data]
  WHERE CONVERT(nvarchar(6), FECHA, 112) >= {}
'''.format(str(reference_time.year)+str(reference_time.month))

df_raw = get_sql_data(query)
df_raw = process_sales(df_raw)

df = df_raw[(df_raw['FAMILIA'].isin(['GASES', 'CO2 NO USAR']))]
df = df[~(df['TIPO_CLIENTE_2'].isin(['HOME CARE']))]
df = df[~df['TIPO_DATOS'].isin(['CO2 Liquido','Nitrogeno Liquido',
                                            'OXIGENO LIQUIDO (COMPET.)','Mercaderias'])]
df_class_new = get_sql_data('''SELECT * FROM [DIGITAL_PH_DEV].[car_py].[mapping_ge_prods]''')
df = df.merge(df_class_new.drop(columns=['TIPO_DATOS']), how='left',on='NOMBRE_PRODUCTO')
df['TIPO_DATOS'] = np.where(df['TIPO_DATOS_NEW'].isna(), df['TIPO_DATOS'], df['TIPO_DATOS_NEW'])
df = df.drop(columns=['TIPO_DATOS_NEW'])

df_f1 = data_step2a_process_idperiod(df, 'NO_CLIENTE', 'TIPO_DATOS', 'periodo', 'CENTRO', 'VTA_RECONOCIDA',
                                     'FECHA_FACTURA', 'VOLUMEN_FACTURADO')

# Step 3
# Calibration set up
config={'monetary_value': 'mean','frequency':'sum','volume': 'mean', 'periodo':'max',
        'max_time_between_tr':'max'}
t_attribute_end = pd.to_datetime(df_f1['periodo'].max(), format='%Y%m') + relativedelta(months=-3)
t_attribute_end = str(t_attribute_end.year) + str(t_attribute_end.month)
df_total_sales = calculate_total_sale_last_12M(df_raw)

client_column = 'NO_CLIENTE'
product_column = 'TIPO_DATOS'
sucursal_column = 'CENTRO'
recover = True

t_df_past_tr_24M = data_step2b_last_transaction(df_f1, 12, t_attribute_end, 12, [client_column,product_column, sucursal_column], 'max').rename(columns={'Purchase_Ref_Old':'Purchase_Ref_Old_24M'})
t_df_past_tr_12M = data_step2b_last_transaction(df_f1, 0, t_attribute_end, 11, [client_column,product_column, sucursal_column] ,'min').rename(columns={'Purchase_Ref_Old':'Purchase_Ref_Old_12M'})
t_df_past_tr = df_f1[[client_column, product_column, sucursal_column]].drop_duplicates()
t_df_past_tr = t_df_past_tr.merge(t_df_past_tr_24M, how='left', left_on = [client_column,product_column, sucursal_column], right_index=True).merge(t_df_past_tr_12M, how='left', left_on = [client_column,product_column, sucursal_column], right_index=True)
t_df_past_tr['Purchase_Ref_Old'] = np.where(t_df_past_tr['Purchase_Ref_Old_24M'].isnull(),t_df_past_tr['Purchase_Ref_Old_12M'],t_df_past_tr['Purchase_Ref_Old_24M'])
t_df_past_tr['Ref_Old'] = np.where(t_df_past_tr['Purchase_Ref_Old_24M'].isnull(),'recent','older')
t_df_past_tr = t_df_past_tr.drop(columns=['Purchase_Ref_Old_12M','Purchase_Ref_Old_24M'])

t_df_f2 = data_step3(df_f1, config, t_attribute_end, 12, client_column, 'periodo', product_column, sucursal_column, t_df_past_tr)
# t_df_f2 = t_df_f2.merge(df_cc, how='left', on=[client_column])
t_df_target = data_step3_target_for_cv(df_f1, df_f1['periodo'].max(), 3, client_column, 'periodo', product_column)

# Production data set_up
df_past_tr_24M = data_step2b_last_transaction(df_f1, 12, df_f1['periodo'].max(), 12, [client_column, product_column, sucursal_column], 'max').rename(columns={'Purchase_Ref_Old':'Purchase_Ref_Old_24M'})
df_past_tr_12M = data_step2b_last_transaction(df_f1, 0, df_f1['periodo'].max(), 11, [client_column, product_column, sucursal_column] ,'min').rename(columns={'Purchase_Ref_Old':'Purchase_Ref_Old_12M'})
df_past_tr = df_f1[[client_column,product_column, sucursal_column]].drop_duplicates()
df_past_tr = df_past_tr.merge(df_past_tr_24M, how='left', left_on = [client_column, product_column, sucursal_column], right_index=True).merge(df_past_tr_12M, how='left', left_on = [client_column,product_column, sucursal_column], right_index=True)
df_past_tr['Purchase_Ref_Old'] = np.where(df_past_tr['Purchase_Ref_Old_24M'].isnull(),df_past_tr['Purchase_Ref_Old_12M'],df_past_tr['Purchase_Ref_Old_24M'])
df_past_tr['Ref_Old'] = np.where(df_past_tr['Purchase_Ref_Old_24M'].isnull(),'recent','older')
df_past_tr = df_past_tr.drop(columns=['Purchase_Ref_Old_12M','Purchase_Ref_Old_24M'])

df_f2 = data_step3(df_f1, config, df_f1['periodo'].max(), 12, client_column, 'periodo', product_column, sucursal_column, df_past_tr)
# df_f2 = df_f2.merge(df_cc, how='left', on=[client_column])
print('Step 3 OK. Dataset length:', len(df_f2))


# Step 4
#Dummies de PAR
df_add = df_raw[['NO_CLIENTE','CENTRO','TIPO_DATOS','TIPO_CLIENTE',
                 'TIPO_CLIENTE_2', 'SECTOR', 'FAMILIA']].drop_duplicates()

# Calibration data
t_df_f2_cf = data_step4a(t_df_f2,True,0.999,True,0.001, ['monetary_value', 'volume'])
# t_df_f2_cf = t_df_f2_cf.merge(df_pd2, how='left', on=product_column)
# t_df_f2_cf = t_df_f2_cf.merge(df_pd3, how='left', on=product_column)
t_df_f2_cf[client_column] = t_df_f2_cf[client_column].astype('int64')
t_df_f3 = t_df_f2_cf.merge(df_add, how='inner', on=[client_column, sucursal_column, product_column])
t_df_f3['data_type'] = 'Training'

# Production data
df_f2_cf = data_step4a(df_f2,True,0.999,True,0.001, ['monetary_value', 'volume'])
# df_f2_cf = df_f2_cf.merge(df_pd2, how='left', on=product_column)
# df_f2_cf = df_f2_cf.merge(df_pd3, how='left', on=product_column)
df_f2_cf[client_column] = df_f2_cf[client_column].astype('int64')
df_f3 = df_f2_cf.merge(df_add, how='inner', on=[client_column, sucursal_column, product_column])
df_f3['data_type'] = 'Production'

df_f3_cj_pre = pd.concat([t_df_f3, df_f3], ignore_index=True)
df_f3_cj = data_step4b(df_f3_cj_pre, [sucursal_column, product_column, 'TIPO_CLIENTE', 'TIPO_CLIENTE_2', 'SECTOR', 'FAMILIA'])
df_f3_cj = df_f3_cj.loc[:,~df_f3_cj.columns.duplicated()]

t_df_f3_2 = df_f3_cj[df_f3_cj['data_type'] == 'Training'].drop(columns=['data_type']).reset_index(drop=True)
df_f3_2 = df_f3_cj[df_f3_cj['data_type'] == 'Production'].drop(columns=['data_type']).reset_index(drop=True)

t_df_f3_2 = t_df_f3_2.merge(t_df_target, how='left', on= [client_column,product_column])
t_df_f3_2['target'] = t_df_f3_2['target'].fillna(1)
t_df_f3_2 = t_df_f3_2.drop(columns=['future_spend_total','future_spend_avg'])

rf_model = calibrate_model(t_df_f3_2)
model_metrics = generate_metrics(t_df_f3_2)
df_f4 = model_and_predict(df_f3_2, rf_model)
print('Step 4 OK. Dataset length:', len(df_f4))

# Step 5
# df_m1 = df_f3[[client_column,product_column]]
df_m2 = df_raw[[client_column,'NOMBRE']].drop_duplicates().groupby(client_column).NOMBRE.first()
# df_f5 = df_f4.merge(df_m1, how='left', on=[client_column,product_column])
df_f5 = df_f4.merge(df_m2, how='left', left_on=client_column, right_index=True)
df_f5 = df_f5[['NOMBRE',client_column,sucursal_column,product_column,'Churn_Probability',
               'periodo_analisis', 'last_purchase','monetary_value','frequency','volume','Mean_time_between_tr',
               'max_time_between_tr']]

# df_f5 = df_f5.drop(columns=['CLIENTE']).rename(columns={'sucursal_documento':'sucursal'}).merge(df_mrg, how='left',
#                                                                                             on=[product_column,'sucursal'])
df_f5 = risk_classification_gen(df_f5)
df_f5['Churn_Probability'] = np.where(df_f5['Risk_Level'] == 'Oportunidad', np.nan, df_f5['Churn_Probability'])
df_f5['Risk_Level'] = np.where(df_f5['periodo_analisis'] == df_f5['last_purchase'],
                               'Compra Rcte', df_f5['Risk_Level'])#NEW


df_f5['margin'] = 0
df_m3 = df_f5.copy()
df_m3['total_monetary_value'] = df_m3['monetary_value']*df_m3['frequency']
print('Step 5 OK. Dataset length:', len(df_f5))

#Step 6
df_c = df_m3[['NOMBRE',client_column]].drop_duplicates()
df_c['Estado'] = ''
df_c['Fecha_planificada'] = ''
df_m4 = df_raw[[client_column,sucursal_column,'TIPO_CLIENTE','TIPO_CLIENTE_2','SECTOR']].drop_duplicates()
df_c = df_c.merge(df_m4, how='left', on=[client_column])
df_c = df_c.merge(df_total_sales.rename(columns={'VTA_RECONOCIDA':'Facturacion12M'})
                  , how='left', on=[client_column,sucursal_column])

df_m3_rk = df_m3[~df_m3['Risk_Level'].isin(['Oportunidad'])]
df_c_rk = df_m3_rk.merge(df_m3_rk.groupby(client_column).agg({'total_monetary_value':'sum'}).rename(
 columns={'total_monetary_value':'client_mv'}), how='left',
            left_on=client_column,right_index=True)
df_c_rk['percentage_fact'] = df_c_rk['total_monetary_value'] / df_c_rk['client_mv']
df_c_rk['client_churn_prob_fact'] = df_c_rk['Churn_Probability'] * df_c_rk['percentage_fact']
df_c_rk['margin_fact'] = df_c_rk['margin'] * df_c_rk['percentage_fact']
df_c_rk_aux = df_c_rk.groupby(client_column).margin_fact.sum().rename('client_margin_est')
df_c_rk = df_c_rk.merge(df_c_rk.groupby(client_column).client_churn_prob_fact.sum().rename('client_churn_probability'), how='left',
                    left_on=client_column,right_index=True).merge(df_c_rk_aux, how='left', left_on=client_column,right_index=True)
df_c_rk = df_c_rk[[client_column,'client_churn_probability','client_margin_est']].drop_duplicates()
df_c = df_c.merge(df_c_rk, how='left', on=client_column)
df_c['expected_loss_mv_on_churn'] = df_c['Facturacion12M'] * df_c['client_churn_probability'] *  df_c['client_margin_est']

df_m3_op = df_m3[df_m3['Risk_Level'] == 'Oportunidad']
df_m3_op['total_monetary_value_adj'] = df_m3_op['total_monetary_value'] * df_m3_op['margin']
df_c_op = df_m3_op.merge(df_m3_op.groupby(client_column).agg({'total_monetary_value_adj':'sum'}).rename(
 columns={'total_monetary_value_adj':'potential_win_opportunity'}), how='left',
            left_on=client_column,right_index=True)
df_c_op = df_c_op[[client_column,'potential_win_opportunity']].drop_duplicates()
df_c = df_c.merge(df_c_op, how='left', on=client_column)
print('C1:',len(df_c))
df_c = df_c.merge(df_raw[['NO_CLIENTE','CENTRO','TELEFONO','EMAIL1','VENDEDOR']].drop_duplicates(),
                  how='left',on=['NO_CLIENTE','CENTRO'])
print('C2:',len(df_c))

df_c['client_churn_probability'] = df_c['client_churn_probability'].fillna(0)
df_c['expected_loss_mv_on_churn'] = df_c['expected_loss_mv_on_churn'].fillna(0)
# df_c['potential_win_opportunity'] = df_c['potential_win_opportunity'].fillna(0)

print('Total de clientes:', len(df_c))
print('Clientes en riesgo:', len(df_c[df_c['client_churn_probability'] >= 0.5]))
print('Clientes en riesgo con oportunidades:', len(df_c[(df_c['client_churn_probability'] >= 0.5) &
                                                        (~df_c['potential_win_opportunity'].isna())]))
print('Clientes fuera de riesgo, pero con oportunidades:', len(df_c[~df_c['potential_win_opportunity'].isna()]))

df_aux_op = df_m3[(df_m3['Risk_Level'] == 'Oportunidad') & 
                  (pd.to_datetime(df_m3['last_purchase'], format='%Y%m') < (pd.to_datetime(df_raw['periodo'].max(), format='%Y%m') + relativedelta(months=-1))) &
                  (pd.to_datetime(df_m3['last_purchase'], format='%Y%m') >= (pd.to_datetime(df_raw['periodo'].max(), format='%Y%m') + relativedelta(months=-4)))][['NOMBRE',client_column,'total_monetary_value','margin']].drop_duplicates()
df_aux_op['total_monetary_value_adj'] = df_aux_op['total_monetary_value'] * df_aux_op['margin']
df_aux_op = df_aux_op.sort_values(by='total_monetary_value_adj', ascending=False).head(500)
df_aux_rk = df_m3[(~df_m3['Risk_Level'].isin(['Oportunidad','Compra Rcte'])) &
                  (df_m3['Churn_Probability'] >= 0.5)][['NOMBRE',
                                                                client_column,'total_monetary_value']].drop_duplicates()

df_c = df_c.loc[:,~df_c.columns.duplicated()]
df_c2 = df_c[(df_c[client_column].isin(df_aux_op[client_column].unique())) | (df_c[client_column].isin(df_aux_rk[client_column].unique()))]
# df_c2 = df_c2[df_c2['VENDEDOR']!='VENDEDOR DEPOT']

df_r = (df_f5[(df_f5[client_column].isin(df_c2[client_column].unique())) &
    (((df_f5['Churn_Probability'] >= 0.5) & (df_f5['Risk_Level'] != 'Compra Rcte')) #NEW
               | (df_f5['Risk_Level'] == 'Oportunidad'))]).drop(columns=['NOMBRE'])
df_c2 = df_c2[df_c2[client_column].isin(df_r[client_column].unique())]

df_c2 = df_c2.sort_values(by='potential_win_opportunity', ascending=False)
df_c2 = df_c2.sort_values(by='client_churn_probability', ascending=False)
df_c2 = df_c2.sort_values(by='expected_loss_mv_on_churn', ascending=False)
df_c2['telefono_alternativo'] = ''
df_c2['nombre_persona_contactada'] = ''
df_c2['motivo_contacto_fallido'] = ''
df_c2['estado_propuesta'] = ''
df_c2['valor_propuesta'] = ''
df_c2['motivo_venta_nocerrada'] = ''
df_c2['observaciones'] = ''
df_c2['fecha_ultima_accion'] = ''

#Segm y filtrado de tabla
df_r = df_f5[df_f5[client_column].isin(df_c2[client_column].unique())]
df_r['Buy_Probability'] = np.where(df_r['Risk_Level'] == 'Oportunidad', '', 1-df_r['Churn_Probability'])

print('Step 6 OK. Prod-Client Dataset length:', len(df_r))
print('           Client list Dataset length:', len(df_c2))

#Recuperar updates ultimo mes
if recover:
    df_c2_old = get_sql_data(''' SELECT * FROM [DIGITAL_PH_DEV].[car_py].[last_month_clientlist]''')
    df_c2_old = df_c2_old[(df_c2_old[client_column].isin(df_c2[client_column].unique())) & (~df_c2_old['Estado'].isna()) &
                          (df_c2_old['motivo_contacto_fallido'] != 'EL CLIENTE NO ESTÁ EN RIESGO')]
    for i in df_c2_old.index:
        df_c2.loc[df_c2[client_column] == df_c2_old.loc[i, client_column], 'Estado'] = df_c2_old.loc[i, 'Estado']
        df_c2.loc[df_c2[client_column] == df_c2_old.loc[i, client_column], 'estado_propuesta'] = df_c2_old.loc[i, 'estado_propuesta']
        df_c2.loc[df_c2[client_column] == df_c2_old.loc[i, client_column], 'Fecha_planificada']=df_c2_old.loc[i, 'Fecha_planificada']
        df_c2.loc[df_c2[client_column] == df_c2_old.loc[i, client_column], 'telefono_alternativo'] = df_c2_old.loc[i, 'telefono_alternativo']
        df_c2.loc[df_c2[client_column] == df_c2_old.loc[i, client_column], 'nombre_persona_contactada'] = df_c2_old.loc[i, 'nombre_persona_contactada']
        df_c2.loc[df_c2[client_column] == df_c2_old.loc[i, client_column], 'motivo_contacto_fallido'] = df_c2_old.loc[i, 'motivo_contacto_fallido']
        df_c2.loc[df_c2[client_column] == df_c2_old.loc[i, client_column], 'valor_propuesta'] = df_c2_old.loc[i, 'valor_propuesta']
        df_c2.loc[df_c2[client_column] == df_c2_old.loc[i, client_column], 'motivo_venta_nocerrada'] = df_c2_old.loc[i, 'motivo_venta_nocerrada']
        df_c2.loc[df_c2[client_column] == df_c2_old.loc[i, client_column], 'observaciones'] = df_c2_old.loc[i, 'observaciones']
        df_c2.loc[df_c2[client_column] == df_c2_old.loc[i, client_column], 'fecha_ultima_accion'] = df_c2_old.loc[i, 'fecha_ultima_accion']

df_c2['Facturacion12M'] = df_c2['Facturacion12M'].round(0).astype('int64')
df_r['Buy_Probability'] = df_r['Buy_Probability'].replace('',0)
df_r['Buy_Probability'] = df_r['Buy_Probability'].astype('float64')
df_r['Buy_Probability'] = df_r['Buy_Probability'].fillna(0)
df_r['Buy_Probability'] = (df_r['Buy_Probability']*100).round(0).astype('int64')
    
print('Step 7. Process ended')

master_data = df.sort_values(by='periodo', ascending=False)[[client_column, product_column, sucursal_column,
        'SECTOR', 'FAMILIA', 'PROVINCIA','LOCALIDAD', 'VENDEDOR']].drop_duplicates().groupby([client_column,
                                                product_column, sucursal_column]).first().reset_index()

df_tableau_rk =  df_f5.merge(master_data,
                             how='left',
                             on=[client_column, product_column, sucursal_column])
df_tableau_rk = df_tableau_rk.merge(df_total_sales, how='left', on=[client_column,sucursal_column])
df_tableau_rk['periodo_analisis'] = pd.to_datetime(df_tableau_rk['periodo_analisis'], format = "%Y%m")
df_tableau_rk['monetary_value'] = round(df_tableau_rk['monetary_value'],2)
df_tableau_rk = df_tableau_rk.rename(columns = {'VTA_RECONOCIDA':'Facturacion12M'})
df_tableau_hist = df_raw.copy()
month = ((pd.to_datetime("today")+relativedelta(months=-1)).month)
year = ((pd.to_datetime("today")+relativedelta(months=-1)).year)
df_tableau_hist = df_tableau_hist[((pd.to_datetime(df_tableau_hist['FECHA'], format='%Y%m%d')).dt.year == year) &
                ((pd.to_datetime(df_tableau_hist['FECHA'], format='%Y%m%d')).dt.month == month)]
df_c3 = df_c2.copy()
df_c3['periodo_analisis']= (pd.to_datetime("today") + relativedelta(months=-1)).strftime('%Y%m')

#Data insert to DB
def insert_data_to_sql(data, table_name, schema_name, mode='append'):
    server = 'BRARJ2DBSQL01' 
    database = 'DIGITAL_PH_DEV' 
    username = 'BRAXCARPH1' 
    password = 'yKla2dJaG4wmVmK3zrJ$'
    quoted = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER=BRARJ2DBSQL01;DATABASE=DIGITAL_PH_DEV;UID='+username+';PWD='+ password)
    engine = create_engine('mssql+pyodbc:///?odbc_connect={}'.format(quoted))
    data.to_sql(table_name, schema=schema_name, con = engine, if_exists=mode, index=False)
    print('Upload of {} to {} schema: {} completed'.format(table_name, database, schema_name))

insert_data_to_sql(df_c3, 'lista_clientes', 'car_py')
insert_data_to_sql(df_r, 'prod_riesgo', 'car_py')
insert_data_to_sql(df_tableau_rk, 'car_Tableau_risk_view', 'car_py')
insert_data_to_sql(df_tableau_hist.drop(columns=['PERIODO']), 'car_Tableau_historic_view', 'car_py')

print('Script execution finished.')