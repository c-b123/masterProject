import pandas as pd
import numpy as np
import json
import pickle
from functools import reduce

company_tickers = [
    'MRK', 'MS', 'MU', 'NVDA', 'EBAY', 'NFLX', 'QQQ', 'VZ', 'DAL',
    'JNJ', 'QCOM', 'PFE', 'GSK', 'SNY', 'MRK.DE', 'GS', 'JPM', 'UBS', 'WFC',
    '000660.KS', 'INTC', 'WDC', 'AMD', 'MSFT', '2357.TW', 'AMZN', 'WMT', 'TGT',
    'HM-B.ST', 'DIS', 'AAPL', 'DJIA', 'SPX', 'IXIC', 'RUT', 'T', 'USM', 'LUMN',
    'FYBR', 'LUV', 'KLM', 'JBLU', 'LHA.DE', 'NVS', 'ABT', 'MDTKF'
]
company_similarity_mapping = {
    'MRK': ['PFE', 'GSK', 'SNY', 'MRK.DE'],
    'MS': ['GS', 'JPM', 'UBS', 'WFC'],
    'MU': ['000660.KS', 'INTC', 'WDC', 'NVDA'],
    'NVDA': ['AMD', 'MSFT', '2357.TW', 'INTC'],
    'EBAY': ['AMZN', 'WMT', 'TGT', 'HM-B.ST'],
    'NFLX': ['DIS', 'AAPL', 'MSFT', 'AMZN'],
    'QQQ': ['DJIA', 'SPX', 'IXIC', 'RUT'],
    'VZ': ['T', 'USM', 'LUMN', 'FYBR'],
    'DAL': ['LUV', 'KLM', 'JBLU', 'LHA.DE'],
    'JNJ': ['NVS', 'ABT', 'PFE', 'GSK'],
    'QCOM': ['MDTKF', 'NVDA', 'INTC', 'AMD'],
}

sep_tokens = [i * -0.01 for i in range(1, len(company_tickers) + 1)]
comp_token = {}
for i in range(len(company_tickers)):
    comp_token[company_tickers[i]] = sep_tokens[i]

comp_stocks = []
comp_names = []
for comp in company_similarity_mapping:
    df_list = []
    path = './Company info/' + comp + '.csv'
    df_list.append(pd.read_csv(path))
    other_companies = company_similarity_mapping[comp]
    comp_names_temp = [comp]
    for oc in other_companies:
        path = './Company info/' + oc + '.csv'
        df_list.append(pd.read_csv(path))
        comp_names_temp.append(oc)

    comp_names.append(comp_names_temp)
    comp_stocks.append(df_list)

comp_kpi_df = []
for idx in range(len(comp_stocks)):

    comp = comp_stocks[idx]
    temp_comp_names = comp_names[idx]
    kpi_dfs = {
        'Open': [],
        'High': [],
        'Low': [],
        'Close': [],
        'Adj Close': [],
        'Volume': []
    }

    for i in range(len(comp)):
        df = comp[i]
        df = df.dropna()

        ###### for each company split its 
        for kpi in kpi_dfs.keys():
            temp_df = df[['Date', kpi]]
            temp_df = temp_df.rename(columns={kpi: kpi + '_' + temp_comp_names[i]})
            kpi_dfs[kpi].append(temp_df)

    for kpi in kpi_dfs:
        list_of_dfs = kpi_dfs[kpi]
        df = list_of_dfs[0]
        for df_ in list_of_dfs[1:]:
            df = df.merge(df_, on='Date', how='inner')

        df = df.drop(columns=['Date'])
        kpi_dfs[kpi] = df.to_json()

    comp_kpi_df.append(kpi_dfs)

for i in range(len(comp_kpi_df)):
    temp_comp_names = comp_names[i]
    temp_dict = comp_kpi_df[i]

    main_comp = temp_comp_names[0]
    with open(main_comp + '.json', 'w') as f:
        json.dump(temp_dict, f)
