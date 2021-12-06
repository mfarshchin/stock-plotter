# This file contains all functions

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import math
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.models import Span, Legend, ColumnDataSource, CustomJSTransform, CrosshairTool
from bokeh.models.tools import HoverTool
from bokeh.models.tickers import FixedTicker
from bokeh.palettes import Category10
output_notebook()



#---------------------------------------------------------------------------- stock history
def getHistory(stock, days=365*5, interval='1d'):
    # this static method obtains the history of the given stock
    ticker = yf.Ticker(stock)
    tmp = ticker.history(start= (datetime.now()-timedelta(days)).date(), interval=interval)
    tmp.index = pd.to_datetime(tmp.index, utc =True)
    if interval != '1d':
        tmp.index = tmp.index.tz_convert('US/Eastern')
    tmp['Avg'] = (tmp['Open']+tmp['Close'])/2
    tmp.reset_index(inplace=True)
    tmp.drop(columns=['Dividends', 'Stock Splits'], inplace=True)
    if 'Date' in tmp.columns:
        tmp = tmp.rename(columns={'Date':'Datetime'})
    return tmp



#---------------------------------------------------------------------------- volume analysis
def Vol_Dist(df_d, bins=30, decay_halftime=0.3):
    df_d_vd = volume_distibution(df_d, bins=bins, decay_halftime=decay_halftime)
    return df_d_vd


def volume_distibution(df, bins=30, decay_halftime=0.3):
    # This method is to compute the volume distribution over prices
    # this will return volume and price distribution
    stock_df = df.copy()
    pmin = stock_df['Avg'].min()
    pmax = stock_df['Avg'].max()
    num_steps = abs(stock_df.shape[0]*decay_halftime)
    decay_coeff = 0.5**(1/num_steps)
    pstep = (pmax-pmin)/bins
    stock_df['rng'] =stock_df['High'] - stock_df['Low']
    price_centers = np.arange(pmin-stock_df['rng'].max()-pstep,pmax+stock_df['rng'].max()+pstep,pstep)
    acc_volume = price_centers*0
    for _, row in stock_df.iterrows():
        volume = row['Volume']
        avg_point = row['Avg']
        low = row['Low']
        high = row['High']
        acc_volume *= decay_coeff
        acc_volume =accumulation(acc_volume, price_centers, volume, avg_point, low, high)
    return pd.DataFrame({'price_centers':price_centers,'volume_dist':acc_volume})


def accumulation(acc_volume, price_centers, volume, avg_point, low, high):
    # this function is to add the volume to the acc_volume at the right position
    stp = (high-low)/6
    # first
    location = [True if x>avg_point-3*stp and x<=avg_point-2*stp else False for x in price_centers]
    if sum(location)>0:
        acc_volume[location] += 0.05/2*volume/sum(location)
    # second
    location = [True if x>avg_point-2*stp and x<=avg_point-1*stp else False for x in price_centers]
    if sum(location)>0:
        acc_volume[location] += 0.27/2*volume/sum(location)
    # third
    location = [True if x>avg_point-1*stp and x<=avg_point-0*stp else False for x in price_centers]
    if sum(location)>0:
        acc_volume[location] += 0.68/2*volume/sum(location)
    # fourth
    location = [True if x>avg_point-0*stp and x<=avg_point+1*stp else False for x in price_centers]
    if sum(location)>0:
        acc_volume[location] += 0.68/2*volume/sum(location)
    # fifth
    location = [True if x>avg_point+1*stp and x<=avg_point+2*stp else False for x in price_centers]
    if sum(location)>0:
        acc_volume[location] += 0.27/2*volume/sum(location)
    # sixth
    location = [True if x>avg_point+2*stp and x<=avg_point+3*stp else False for x in price_centers]
    if sum(location)>0:
        acc_volume[location] += 0.05/2*volume/sum(location)
    return acc_volume

#---------------------------------------------------------------------------- moving average
def mySimpleMA(tmp,spans=[5,10,21,60]):
    df_ma = tmp.copy()
    for days in spans:
        df_ma[str(days)] = df_ma['Avg'].rolling(window=days).mean()
    return df_ma[['Datetime']+[str(x) for x in spans]]





#---------------------------------------------------------------------------- price volume plot
def plot_price_vol(df, df_v, stock):
    df_v['volume_dist'] = df_v['volume_dist']/df_v['volume_dist'].max()*df.shape[0]*1/3
    tmp = df.copy()
    min_p, max_prc = tmp.Avg.min(), tmp.Avg.max(); rng = max_prc-min_p; mult = 0.25*rng
    tmp.Volume /= tmp.Volume.max()
    lowst = min_p-rng/4
    tmp = tmp.assign(tooltip=[x.strftime('%Y/%m/%d, %H:%M') for x in tmp['Datetime']])
    p = figure(plot_width=1600, plot_height=600, title=f'Stock name: "{stock["name"]}"', background_fill_color="#fafafa", tools='box_zoom,wheel_zoom,pan,reset,save',active_drag='box_zoom',  toolbar_location="above", y_range=(lowst, max_prc+0.1*rng))
    source = ColumnDataSource(data=dict(Index=tmp.index.tolist(), Datetime=tmp['Datetime'].tolist(), prc=tmp['Avg'].tolist(),\
                                        vol=tmp['Volume'].tolist(), tooltip=tmp['tooltip'].tolist()))
    source0 = ColumnDataSource(data=dict(price_centers=df_v['price_centers'].tolist(), volume_dist=df_v['volume_dist'].tolist()))
    # add horizontal line at latest price
    p.add_layout(Span(location=df.Avg.iloc[-1], dimension='width', line_color='magenta', line_width=0.5, line_dash='solid'))
    # volume distribution
    above = df_v['price_centers'] > df.Avg.iloc[-1]
    below = df_v['price_centers'] <= df.Avg.iloc[-1]
    barWidth = (df_v.loc[1,'price_centers']-df_v.loc[0,'price_centers'])*0.9
    p.hbar(y='price_centers', right='volume_dist',height=barWidth*0.9, source=source0, fill_color='green', line_color='green', line_alpha=0.1, fill_alpha=0.1, name='zero')
    p.hbar(y=df_v.price_centers[above], right=df_v.volume_dist[above],height=barWidth, fill_color='red', line_color='red', line_alpha=0.4, fill_alpha=0.4, name='above')
    p.hbar(y=df_v.price_centers[below], right=df_v.volume_dist[below],height=barWidth, fill_color='green', line_color='green', line_alpha=0.4, fill_alpha=0.4, name='below')
    # Average price line
    p.line(x='Index', y='prc', source=source, legend_label="price", color='grey', line_width=1, name='price')
    # candlesticks
    inc = df.Close > df.Open
    dec = df.Open > df.Close
    p.segment(tmp.index[inc], tmp.High[inc], tmp.index[inc], tmp.Low[inc], color="green")
    p.segment(tmp.index[dec], tmp.High[dec], tmp.index[dec], tmp.Low[dec], color="red")
    p.vbar(tmp.index[inc], 0.5, tmp.Open[inc], tmp.Close[inc], fill_color="green", line_color="green")
    p.vbar(tmp.index[dec], 0.5, tmp.Open[dec], tmp.Close[dec], fill_color="red", line_color="red")
    # volume
    p.vbar(tmp.index[inc], 0.5, lowst, lowst+tmp.Volume[inc]*mult, fill_color="green", line_color="green", line_alpha=0.3, fill_alpha=0.3)
    p.vbar(tmp.index[dec], 0.5, lowst, lowst+tmp.Volume[dec]*mult, fill_color="red", line_color="red", line_alpha=0.3, fill_alpha=0.3)
    # hover tools
    hover1 = HoverTool(tooltips=[('Time','@tooltip'), ('Price','$@prc'), ('y','$$y')], names = ['price'], mode = 'vline')
    p.tools.append(hover1)
    hover2 = HoverTool(tooltips=[('Price','$@price_centers')], names = ['zero'], mode = 'hline')
    p.tools.append(hover2)
    # add ticks 
    timestep = (df.Datetime.iloc[10]-df.Datetime.iloc[9])
    p = MonthTicks(tmp, df, p)
    p.legend.location = "top_left"
    p.legend.level = 'overlay'
    p.legend.border_line_width = 2
    p.legend.border_line_color = "grey"
    p.legend.label_text_font_size = '11pt'
    p.title.text_font_size = '14pt'
    p.title.align = "center"
    p.legend.visible = True        
    return p


#---------------------------------------------------------------------------- MA plot
def plot_add_MA(df_ma, p, line_dash='solid'):
    tmp = df_ma.copy()
    # Average price line
    for i, j in enumerate(tmp.columns[1:]):
        p.line(tmp.index, tmp[j], legend_label=f'MA-{j}', color=Category10[10][i], line_width=1, line_dash=line_dash, name='mov')
    return p

#---------------------------------------------------------------------------- purchase price
def plot_add_purchasePrice(purchase_price, p, line_dash='solid'):
    p.add_layout(Span(location=purchase_price, dimension='width', line_color='blue', line_width=1, line_dash=line_dash))
    p.line([], [], legend_label='purcahse price', line_dash=line_dash, line_color="blue", line_alpha=1)
    return p



#----------------------------------------------------------------------------
def MonthTicks(tmp, df, p):
    stockHist = df.copy()
    # sample at monthly
    stockHist.index = stockHist['Datetime']
    monthly = stockHist.resample('m').min().dropna()
    dicti = {}
    FirstOfMonthTicks=[]
    FirstOfYearTicks=[]
    for i, date in enumerate(tmp['Datetime']):
        if (tmp['Datetime'][i] in monthly['Datetime'].tolist()):
            if tmp['Datetime'][i].month==1:
                dicti.update({i:date.strftime('%Y/%m/%d')})
                FirstOfYearTicks.append(i)
            else:
                dicti.update({i:date.strftime('%m/%d')})
            FirstOfMonthTicks.append(i)
    # Add tick lines at month
    for i in FirstOfMonthTicks:
        VLine = Span(location=i, line_dash='dashed',
                     dimension='height', line_color='grey',
                     line_width=1)
        p.add_layout(VLine)
    # Add tick lines at year
    for i in FirstOfYearTicks:
        VLine = Span(location=i, line_dash='dashed',
                     dimension='height', line_color='red',
                     line_width=1)
        p.add_layout(VLine)
    p.xaxis.major_label_overrides = dicti
    p.xaxis.major_label_orientation = math.pi/2
    p.xaxis.ticker = FixedTicker(ticks=FirstOfMonthTicks)
    return p
