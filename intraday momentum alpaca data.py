import pytz
import math
import requests
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from   matplotlib.ticker import FuncFormatter
from   datetime import datetime, timedelta
from   zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.stock import StockDataStream

from alpaca.data.requests import (
    CorporateActionsRequest,
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
)
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetAssetsRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.enums import (
    AssetExchange,
    AssetStatus,
    OrderClass,
    OrderSide,
    OrderType,
    QueryOrderStatus,
    TimeInForce,
)



api_key = ''
secret_key = ''

trade_api_url = None
trade_api_wss = None
data_api_url = None
stream_data_wss = None





def get_intraday_minute_bar(client, ticker, start_date, end_date):
    """
    Alpaca에서 1분봉(정규장) 데이터 호출
    - client: StockHistoricalDataClient 객체
    - ticker: 'QQQ' 등 티커 문자열
    - start_date, end_date: datetime (America/New_York tz-aware)
    """
    # 1분봉 요청 객체 생성
    req = StockBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start_date,
        end=end_date,
    )
    df = client.get_stock_bars(req).df.reset_index()
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
    
    # 정규장(09:30~15:59)만 필터링
    hour = df['timestamp'].dt.hour
    minute = df['timestamp'].dt.minute
    mask = ((hour > 9) & (hour < 15)) | \
           ((hour == 9) & (minute >= 30)) | \
           ((hour == 15) & (minute <= 59))
    df_intra = df[mask].reset_index(drop=True)
    df_intra = df_intra.rename(columns={'timestamp': 'caldt'})
    return df_intra

def get_daily_bar(client, ticker, start_date, end_date):
    """
    Alpaca에서 일봉 데이터 호출
    - client: StockHistoricalDataClient 객체
    - ticker: 'QQQ' 등 티커 문자열
    - start_date, end_date: datetime (America/New_York tz-aware)
    """
    req = StockBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=start_date,
        end=end_date,
    )
    df = client.get_stock_bars(req).df.reset_index()
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_daily = df.rename(columns={'timestamp': 'caldt'})
    return df_daily

def get_dividend_data(corporate_actions_client, ticker, start_date):
    """
    Alpaca에서 배당(cash dividend) 데이터 호출
    - corporate_actions_client: CorporateActionsClient 객체
    - ticker: 'QQQ' 등 티커 문자열
    - start_date: datetime, 배당 데이터 시작 기준일 (끝 날짜는 필요 없음)
    """
    start_date_naive = start_date.replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
    
    req = CorporateActionsRequest(
        start=start_date_naive,
        symbols=[ticker]
    )
    df_div = corporate_actions_client.get_corporate_actions(req).df.reset_index(drop=True)
    return df_div


ticker = 'TQQQ'
start_date = datetime(2020, 1, 1, tzinfo=ZoneInfo("America/New_York")).replace(hour=9, minute=30, second=0)
end_date = (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)).replace(hour=15, minute=59, second=0)

stock_historical_data_client = StockHistoricalDataClient(api_key, secret_key, url_override = data_api_url)
corporate_actions_client = CorporateActionsClient(api_key, secret_key, url_override=data_api_url)

df_1min = get_intraday_minute_bar(stock_historical_data_client, ticker, start_date, end_date)   # 1분봉 데이터
df_daily = get_daily_bar(stock_historical_data_client, ticker, start_date, end_date)            # 1일봉 데이터
df_dividend = get_dividend_data(corporate_actions_client, ticker, start_date)                   # 배당 데이터



# %%

# Load the intraday data into a DataFrame and set the datetime column as the index.
df = pd.DataFrame(df_1min)
df['day'] = pd.to_datetime(df['caldt']).dt.date  # Extract the date part from the datetime for daily analysis.
df.set_index('caldt', inplace=True)  # Setting the datetime as the index for easier time series manipulation.

# Group the DataFrame by the 'day' colum`n to facilitate operations that need daily aggregation.
daily_groups = df.groupby('day')

# Extract unique days from the dataset to iterate through each day for processing.
all_days = df['day'].unique()

# Initialize new columns to store calculated metrics, starting with NaN for absence of initial values.
df['move_open'] = np.nan  # To record the absolute daily change from the open price
df['vwap'] = np.nan       # To calculate the Volume Weighted Average Price.
df['spy_dvol'] = np.nan   # To record SPY's daily volatility.

# Create a series to hold computed daily returns for SPY, initialized with NaN.
spy_ret = pd.Series(index=all_days, dtype=float)

# Iterate through each day to calculate metrics.
for d in range(1, len(all_days)):
    current_day = all_days[d]
    prev_day = all_days[d - 1]
    
    # Access the data for the current and previous days using their groups.
    current_day_data = daily_groups.get_group(current_day)
    prev_day_data = daily_groups.get_group(prev_day)

    # Calculate the average of high, low, and close prices.
    hlc = (current_day_data['high'] + current_day_data['low'] + current_day_data['close']) / 3

    # Compute volume-weighted metrics for VWAP calculation.
    vol_x_hlc = current_day_data['volume'] * hlc
    cum_vol_x_hlc = vol_x_hlc.cumsum()  # Cumulative sum for VWAP calculation.
    cum_volume = current_day_data['volume'].cumsum()

    # Assign the calculated VWAP to the corresponding index in the DataFrame.
    df.loc[current_day_data.index, 'vwap'] = cum_vol_x_hlc / cum_volume

    # Calculate the absolute percentage change from the day's opening price.
    open_price = current_day_data['open'].iloc[0]
    df.loc[current_day_data.index, 'move_open'] = (current_day_data['close'] / open_price - 1).abs()

    # Compute the daily return for SPY using the closing prices from the current and previous day.
    spy_ret.loc[current_day] = current_day_data['close'].iloc[-1] / prev_day_data['close'].iloc[-1] - 1

    # Calculate the 15-day rolling volatility, starting calculation after accumulating 15 days of data.
    if d > 14:
        df.loc[current_day_data.index, 'spy_dvol'] = spy_ret.iloc[d - 15:d - 1].std(skipna=False)
        
    # break


# Calculate the minutes from market open and determine the minute of the day for each timestamp.
df['min_from_open'] = ((df.index - df.index.normalize()) / pd.Timedelta(minutes=1)) - (9 * 60 + 30) + 1
df['minute_of_day'] = df['min_from_open'].round().astype(int)

# Group data by 'minute_of_day' for minute-level calculations.
minute_groups = df.groupby('minute_of_day')

# Calculate rolling mean and delayed sigma for each minute of the trading day.
df['move_open_rolling_mean'] = minute_groups['move_open'].transform(lambda x: x.rolling(window=14, min_periods=13).mean())
df['sigma_open'] = minute_groups['move_open_rolling_mean'].transform(lambda x: x.shift(1))

# Convert dividend dates to datetime and merge dividend data based on trading days.
df_dividend['day'] = pd.to_datetime(df_dividend['ex_date']).dt.date
df = df.merge(df_dividend[['day', 'rate']], on='day', how='left')
df = df.rename(columns={'rate':'dividend'})
df['dividend'] = df['dividend'].fillna(0)  # Fill missing dividend data with 0.



# %%


# Constants and settings
AUM_0 = 100000.0
commission = 0.0035
min_comm_per_order = 0.35
band_mult = 1
band_simplified = 0
trade_freq = 30
sizing_type = "full_notional"
target_vol = 0.02
max_leverage = 1


# Group data by day for faster access
daily_groups = df.groupby('day')

# Initialize strategy DataFrame using unique days
strat = pd.DataFrame(index=all_days)
strat['ret'] = np.nan
strat['AUM'] = AUM_0
strat['ret_spy'] = np.nan

# Calculate daily returns for SPY using the closing prices
# df_daily = pd.DataFrame(daily_data)
df_daily['caldt'] = pd.to_datetime(df_daily['caldt']).dt.date
df_daily.set_index('caldt', inplace=True)  # Set the datetime column as the DataFrame index for easy time series manipulation.
df_daily['ret'] = df_daily['close'].diff() / df_daily['close'].shift()


# Loop through all days, starting from the second day
for d in range(1, len(all_days)):
    current_day = all_days[d]
    prev_day = all_days[d-1]
    
    if prev_day in daily_groups.groups and current_day in daily_groups.groups:
        prev_day_data = daily_groups.get_group(prev_day)
        current_day_data = daily_groups.get_group(current_day)

        if 'sigma_open' in current_day_data.columns and current_day_data['sigma_open'].isna().all():
            continue

        prev_close_adjusted = prev_day_data['close'].iloc[-1] - df.loc[current_day_data.index, 'dividend'].iloc[-1]

        open_price = current_day_data['open'].iloc[0]
        current_close_prices = current_day_data['close']
        spx_vol = current_day_data['spy_dvol'].iloc[0]
        vwap = current_day_data['vwap']

        sigma_open = current_day_data['sigma_open']
        UB = max(open_price, prev_close_adjusted) * (1 + band_mult * sigma_open)
        LB = min(open_price, prev_close_adjusted) * (1 - band_mult * sigma_open)

        # Determine trading signals
        signals = np.zeros_like(current_close_prices)
        signals[(current_close_prices > UB) & (current_close_prices > vwap)] = 1
        signals[(current_close_prices < LB) & (current_close_prices < vwap)] = -1

        # Position sizing
        previous_aum = strat.loc[prev_day, 'AUM']

        if sizing_type == "vol_target":
            if math.isnan(spx_vol):
                shares = round(previous_aum / open_price * max_leverage)
            else:
                shares = round(previous_aum / open_price * min(target_vol / spx_vol, max_leverage))

        elif sizing_type == "full_notional":
            shares = round(previous_aum / open_price)

        # Apply trading signals at trade frequencies
        trade_indices = np.where(current_day_data["min_from_open"] % trade_freq == 0)[0]
        exposure = np.full(len(current_day_data), np.nan)  # Start with NaNs
        exposure[trade_indices] = signals[trade_indices]  # Apply signals at trade times

        # Custom forward-fill that stops at zeros
        last_valid = np.nan  # Initialize last valid value as NaN
        filled_values = []  # List to hold the forward-filled values
        for value in exposure:
            if not np.isnan(value):  # If current value is not NaN, update last valid value
                last_valid = value
            if last_valid == 0:  # Reset if last valid value is zero
                last_valid = np.nan
            filled_values.append(last_valid)

        exposure = pd.Series(filled_values, index=current_day_data.index).shift(1).fillna(0).values  # Apply shift and fill NaNs

        # Calculate trades count based on changes in exposure
        trades_count = np.sum(np.abs(np.diff(np.append(exposure, 0))))

        # Calculate PnL
        change_1m = current_close_prices.diff()
        gross_pnl = np.sum(exposure * change_1m) * shares
        commission_paid = trades_count * max(min_comm_per_order, commission * shares)
        net_pnl = gross_pnl - commission_paid

        # Update the daily return and new AUM
        strat.loc[current_day, 'AUM'] = previous_aum + net_pnl
        strat.loc[current_day, 'ret'] = net_pnl / previous_aum

        # Save the passive Buy&Hold daily return for SPY
        strat.loc[current_day, 'ret_spy'] = df_daily.loc[df_daily.index == current_day, 'ret'].values[0]
        
        
# %%

# Calculate cumulative products for AUM calculations
strat['AUM_SPX'] = AUM_0 * (1 + strat['ret_spy']).cumprod(skipna=True)

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Plotting the AUM of the strategy and the passive S&P 500 exposure
ax.plot(strat.index, strat['AUM'], label='Momentum', linewidth=2, color='k')
ax.plot(strat.index, strat['AUM_SPX'], label='S&P 500', linewidth=1, color='r')

# Formatting the plot
ax.grid(True, linestyle=':')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
plt.xticks(rotation=90)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_ylabel('AUM ($)')
plt.legend(loc='upper left')
plt.title('Intraday Momentum Strategy', fontsize=12, fontweight='bold')
plt.suptitle(f'Commission = ${commission}/share', fontsize=9, verticalalignment='top')

# Show the plot
plt.show()

# Calculate additional stats and display them
stats = {
    'Total Return (%)': round((np.prod(1 + strat['ret'].dropna()) - 1) * 100, 0),
    'Annualized Return (%)': round((np.prod(1 + strat['ret']) ** (252 / len(strat['ret'])) - 1) * 100, 1),
    'Annualized Volatility (%)': round(strat['ret'].dropna().std() * np.sqrt(252) * 100, 1),
    'Sharpe Ratio': round(strat['ret'].dropna().mean() / strat['ret'].dropna().std() * np.sqrt(252), 2),
    'Hit Ratio (%)': round((strat['ret'] > 0).sum() / (strat['ret'].abs() > 0).sum() * 100, 0),
    'Maximum Drawdown (%)': round(strat['AUM'].div(strat['AUM'].cummax()).sub(1).min() * -100, 0)
}


Y = strat['ret'].dropna()
X = sm.add_constant(strat['ret_spy'].dropna())
model = sm.OLS(Y, X).fit()
stats['Alpha (%)'] = round(model.params.const * 100 * 252, 2)
stats['Beta'] = round(model.params['ret_spy'], 2)

print(stats)




# %%
# real time market data
# Compact Alpaca Stream
import os, sys, signal, threading, time
from collections import defaultdict
from datetime import datetime, timezone, timedelta

# 실시간 스트림
from alpaca.data.enums import DataFeed
from alpaca.data.live.crypto import CryptoDataStream
from alpaca.data.live.stock import StockDataStream

# 시간대 설정
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:
    ET = timezone.utc  # tzdata 미설치 시 UTC로 폴백(표기만 달라짐)



API_KEY      = os.getenv("APCA_API_KEY_ID", "")
API_SECRET   = os.getenv("APCA_API_SECRET_KEY", "")

USE_CRYPTO       = True                # True=크립토, False=주식
SYMBOLS_CRYPTO   = ["BTC/USD"]
SYMBOLS_STOCK    = ["QQQ"]
FEED             = DataFeed.IEX        # 주식 전용: IEX(무료) / SIP(유료)



# -------------------------------------------------------------
# UTC & ET 동시 문자열
# -------------------------------------------------------------
def utc_et_str(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    u = ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    e = ts.astimezone(ET).strftime("%Y-%m-%d %H:%M:%S")
    label = "ET" if ET is not timezone.utc else "UTC"
    return f"{u} UTC | {e} {label}"

# -------------------------------------------------------------
# 콜백: 1분봉 수신 시 호출 (크립토/주식 공통 형식)
# -------------------------------------------------------------
async def on_bar(bar):
    ts = getattr(bar, "timestamp", None) or getattr(bar, "t", None)
    if isinstance(ts, datetime):
        ts_str = utc_et_str(ts)
    else:
        ts_str = str(ts)

    vwap = getattr(bar, "vwap", None) or getattr(bar, "vw", None)
    vol  = getattr(bar, "volume", None) or getattr(bar, "v", None)
    vwap_s = f"{vwap:.6f}" if isinstance(vwap, (int, float)) else "-"
    vol_s  = f"{vol:.6f}"  if isinstance(vol,  (int, float)) else "-"

    print(
        f"[{ts_str}] {bar.symbol} "
        f"O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} "
        f"V:{vol_s} VWAP:{vwap_s}",
        flush=True
    )

# -------------------------------------------------------------
# 실행
# -------------------------------------------------------------
def main():
    if not API_KEY or not API_SECRET or "YOUR_" in API_KEY or "YOUR_" in API_SECRET:
        print("환경변수 APCA_API_KEY_ID / APCA_API_SECRET_KEY 를 설정하세요.")
        sys.exit(1)

    if USE_CRYPTO:
        stream = CryptoDataStream(API_KEY, API_SECRET)
        symbols = SYMBOLS_CRYPTO
        print(f"Starting CRYPTO 1-minute bars: {', '.join(symbols)}", flush=True)
    else:
        stream = StockDataStream(API_KEY, API_SECRET, feed=FEED)
        symbols = SYMBOLS_STOCK
        print(f"Starting STOCK 1-minute bars (feed={FEED.name}): {', '.join(symbols)}", flush=True)

    stream.subscribe_bars(on_bar, *symbols)

    def shutdown(*_): # 종료(CTRL+C) 처리
        try:
            stream.stop()
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    stream.run()

if __name__ == "__main__":
    main()
