import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt


def draw_kindle(df):
    my_color = mpf.make_marketcolors(up='red',
                                     down='green',
                                     edge='inherit',
                                     wick='inherit')

    # 设置图表的背景色
    my_style = mpf.make_mpf_style(marketcolors=my_color,
                                  figcolor='(0.82, 0.83, 0.85)',
                                  gridcolor='(0.82, 0.83, 0.85)')

    histogram_positive = pd.DataFrame([i if i > 0 else None for i in df['macd'].values])
    histogram_negative = pd.DataFrame([i if i < 0 else None for i in df['macd'].values])
    vol_positive = pd.DataFrame(
        [row['volume'] if row['change'] >= 0 else None for _, row in df.iterrows()])
    vol_negative = pd.DataFrame(
        [row['volume'] if row['change'] < 0 else None for _, row in df.iterrows()])
    add_plot = [
        # MA
        mpf.make_addplot(df['boll_mid'], type='line', color='black', width=0.6),
        mpf.make_addplot(df['MA10'], type='line', color='yellow', width=0.6),
        mpf.make_addplot(df['MA20'], type='line', color='fuchsia', width=0.6),

        # 交易量
        mpf.make_addplot(vol_positive, type='bar', width=0.7, panel=1, color='red'),
        mpf.make_addplot(vol_negative, type='bar', width=0.7, panel=1, color='green'),

        # macd
        mpf.make_addplot(histogram_positive, type='bar', width=0.7, panel=2, color='red'),  # 4
        mpf.make_addplot(histogram_negative, type='bar', width=0.7, panel=2, color='green'),
        mpf.make_addplot(df['dif'], panel=2, color='black', secondary_y=True, type='line', width=0.6),  # 5
        mpf.make_addplot(df['dea'], panel=2, color='yellow', secondary_y=True, type='line', width=0.6),
        # kdj
        mpf.make_addplot(df['k'], type='line', panel=3, color='black', width=0.6),
        mpf.make_addplot(df['d'], type='line', panel=3, color='yellow', width=0.6),
        mpf.make_addplot(df['j'], type='line', panel=3, color='fuchsia', width=0.6)  # 7

    ]
    fig, axes = mpf.plot(df, type="candle", addplot=add_plot, title="Candlestick", ylabel="price($)", style=my_style,
                         main_panel=0,
                         panel_ratios=(1, 0.5, 1, 1), figratio=(12, 8), returnfig=True)
    axes[0].legend(['boll_mid', 'MA10', "MA20"], loc='upper right')
    axes[5].legend(['dif', 'dea'], loc='upper left')
    axes[6].legend(['k', 'd'], loc='upper left')
    axes[7].legend(['j'], loc='lower left')
    plt.savefig("./test.jpg")

if __name__ == "__main__":
    hist = pd.read_csv("../dataset/history_data.csv")
    hist = hist[hist['trade_date'] < 20200401]
    hist.index = pd.to_datetime(hist['trade_date'])
    hist = hist.groupby('ts_code')
    groups = list(hist.groups.keys())
    for i in range(len(groups)):
        stock_data = hist.get_group(groups[i])
        draw_kindle(stock_data)
        if i == 1:
            break