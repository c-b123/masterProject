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
