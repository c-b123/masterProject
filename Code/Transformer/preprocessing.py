'''

Take CSV file containing stock prices and convert to numpy array 
Numpy array is saved as json file 



'''
import pandas as pd
import numpy as np
import json


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


file_name = '../archive/all_stocks_5yr.csv'
df = pd.read_csv(file_name)

df = df.dropna()

KPIs = ['open', 'high', 'low', 'close', 'volume']
sample_tensor = df[[KPIs[0], 'Name', 'date']]

# JSON file to save;
# list of kpis mapping to numpy arrays such that each array has company stock prices
dataset = {}

company_name = []
x = sample_tensor.groupby(['Name'])
for name, x in x:
    if len(x) == 1259:
        company_name.append(name)

for kpi in KPIs:

    print('\n\nKPI: ', kpi)
    sample_tensor = df[[kpi, 'Name', 'date']]
    x = sample_tensor.groupby(['Name'])
    company_dataframes = [d for _, d in x]

    ####### GET LENGTHS ##########
    lengths = {}
    for comp in company_dataframes:
        c = len(comp)
        try:
            lengths[c] = lengths[c] + 1
        except:
            lengths[c] = 1
    print('length: ', lengths)

    ####### GET COMPANIES ##########
    company_sep_token = {}
    i = 0.02
    for name, stock in x:
        if len(stock) == max(lengths):
            company_sep_token[name] = -1 * i
            i += 0.02
    print(company_sep_token)

    ####### GET KPI MATRIX ##########

    kpi_matrix = np.empty([0, max(lengths)])
    for company in company_dataframes:
        if len(company) != max(lengths):
            continue
        kpi_matrix = np.vstack([kpi_matrix, company[kpi].to_numpy()])

    dataset[kpi] = {
        'data': kpi_matrix,
        'company': company_sep_token
    }

dumped = json.dumps(dataset, default=default)
with open('processed_stock_dataset.json', 'w') as f:
    json.dump(dumped, f, indent=6)
