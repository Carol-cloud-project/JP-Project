import pandas as pd
import yfinance as yf

# # 从Wikipedia获取标普500公司列表
# url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# table = pd.read_html(url)[0]  # 读取第一个表格


# # 获取股票代码列表
# tickers = table['Symbol'].tolist()
# print('read 1')

# url = "https://en.wikipedia.org/wiki/NASDAQ-100"
# table = pd.read_html(url)[4]  # 纳斯达克100表格通常是第5个表格（索引是4）

# # 提取股票代码
# tickers_NASDAQ = table['Symbol'].tolist()

# # 从Wikipedia获取DJIA成分股列表
# url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
# table = pd.read_html(url)[2]  # 通常DJIA成分股表格是第3个表格

# # 获取股票代码列表
# tickers_DJIA = table['Symbol'].tolist()

# list1=list(set(tickers+tickers_NASDAQ+tickers_DJIA))

# Russell=pd.read_csv('/home/rliuaj/balance_sheet/Russell_3000_Tickers.csv')
# tickers_Russell = Russell['Ticker'].tolist()

# list2=[item for item in tickers_Russell if item not in list1]

# stock_list=list1+list2
stock=pd.read_csv('stock_list.csv')
stock_list=stock['stock']

df_X = pd.DataFrame()

df_y = pd.DataFrame()
count=1
for name in stock_list:
    try:
        # Assuming MEDC.financials and MEDC.balancesheet are already loaded as pandas DataFrames
        company=yf.Ticker(f'{name}')#("MEDC.JK")

        balancesheet_selected = company.balancesheet.loc[
            ['Current Assets','Total Non Current Assets',
        'Current Liabilities','Total Non Current Liabilities Net Minority Interest',
        'Stockholders Equity']
        ]

        # # Step 3: Get the intersection of columns in both dataframes
        # common_columns = financials_selected.columns.intersection(balancesheet_selected.columns)

        # # Step 4: Concatenate the two rows with common columns
        # result = pd.concat([financials_selected[common_columns], balancesheet_selected[common_columns]], axis=0)

        # # Display the result
        # # 创建 DataFrame
        # df_m = pd.DataFrame(result, index=[
        # 'Current Assets','Total Non Current Assets',
        # 'Current Liabilities','Total Non Current Liabilities Net Minority Interest',
        # 'Stockholders Equity'
        # ])

        # 步骤 1: 去除包含超过50% NaN 的列
        threshold = len(balancesheet_selected) * 0.5
        df = balancesheet_selected.dropna(axis=1, thresh=threshold)



        # 获取公司基本信息
        info = company.info

        # 提取所需信息
        market_cap = info.get('marketCap')
        industry = info.get('industry')
        employees = info.get('fullTimeEmployees')
        city = info.get('city')
        ebitda_margins = info.get('ebitdaMargins')
        profit_margins = info.get('profitMargins')
        gross_margins = info.get('grossMargins')
        day_high = info.get('dayHigh')

        # 添加新行
        new_rows = {
            'Market Cap': market_cap,
            'Industry': industry,
            'Employees': employees,
            'City': city,
            'EBITDA Margins': ebitda_margins,
            'Profit Margins': profit_margins,
            'Gross Margins': gross_margins,
            'Day High': day_high
        }

        # 对于每一个新行信息，我们将其添加到 DataFrame
        for key, value in new_rows.items():
            df.loc[key] = pd.Series([value] * len(df.columns), index=df.columns)


        # 定义特征和标签

        features_columns = df.index.tolist()  # 所有列作为特征
        target_columns = ['Current Assets','Total Non Current Assets',
        'Current Liabilities','Total Non Current Liabilities Net Minority Interest',
        'Stockholders Equity']  # 特定列作为标签

        # 构建特征和标签数据集
        X = df.T[features_columns][1:]  # 使用前一年数据作为特征
        y = df.T[target_columns][:-1]  # 当前年数据作为标签
        df_X=pd.concat([df_X, X])#, ignore_index=True)
        df_y=pd.concat([df_y, y])

        print('add')

        count+=1
        if count%2==0:
            df_X.to_csv('df_X')
            df_y.to_csv('df_y')
            # 将数字保存到DataFrame
            df_num = pd.DataFrame([count], columns=["Number"])

            # 将DataFrame保存为CSV文件
            df_num.to_csv("count.csv", index=False)
    except:
        pass
