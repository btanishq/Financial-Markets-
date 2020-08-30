import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import mplfinance as mpf
import matplotlib.dates as mdates


style.use('ggplot')
start = dt.datetime(2000,1,1)
end = dt.datetime(2016,12,31)

# df = web.DataReader('TSLA', 'yahoo', start, end)
#df.to_csv('tsla.csv')


df = pd.read_csv('tsla.csv', parse_dates = True, index_col= 0)

df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

mpf.plot(df, type='candle', style='charles',volume=True)
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1,colspan=1, sharex=ax1)


# #print(df.head)
# print(df[['Open','High']].head())
# df['Adj Close'].plot()
# plt.show()

#The new version of mplfinance as of Feb 2020 allows this whole video to be done with 3 lines:


# import mplfinance as mpf
# df= pd.read_csv('IBM.csv',parse_dates=True, index_col=0)
# mpf.plot(df, type='candle', style='charles',
#             title='  ',
#             ylabel='  ',
#             ylabel_lower='  ',
#             figratio=(25,10),
#             figscale=1,
#             mav=50,
#             volume=True
#             )