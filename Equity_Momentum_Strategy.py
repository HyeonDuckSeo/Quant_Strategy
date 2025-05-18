# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xlwings as xw
import pickle
import seaborn as sns
from IPython.display import display, HTML

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 4)


# prices data
URL = 'C:/Users/서현덕SGAM글로벌멀티투자본부알고리즘/Desktop/Quant_Strategy-main/Momentum/prices_data.xlsx'
prices = pd.read_excel(URL,index_col=0)
prices.ffill(inplace=True)

prices_equity = prices[['S&P500', 'MSCI EM']]
prices_riskoff = prices[['LBMA Gold Price', 'Bloomberg US Long Treasury Total Return Index Value Unhedged']]

prices_equity = prices_equity.dropna()
prices_riskoff = prices_riskoff.dropna()


# 기초자산 상관관계
merged_prices = pd.merge(prices_equity, prices_riskoff, left_index=True, right_index=True, how='inner')
log_returns = np.log(merged_prices / merged_prices.shift(1)).dropna()
corr_matrix = log_returns.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Log Return Correlation Heatmap (Equity vs Risk-Off Assets)")
plt.tight_layout()
plt.show()





# %%
# Strategy Backtesting
class momentum_strategy:
    def __init__(self, prices, lookback, n_selection, signal_method, cost, long_only, start_date, end_date, monthly_data):
        
        self.returns = self.get_returns(prices)
        self.monthly_prices = self.get_monthly_prices(prices)
        self.monthly_returns = self.get_monthly_returns(self.monthly_prices)
        
        if signal_method == 'absolute_momentum':
            self.signal, self.lookback_returns = self.absolute_momentum(self.monthly_prices, lookback, long_only)
        elif signal_method == 'relative_momentum':
            self.signal, self.lookback_returns, self.rank = self.relative_momentum(self.monthly_prices, lookback, n_selection, long_only)
        elif signal_method == 'dual_momentum':
            self.signal, self.abs_signal, self.rel_signal, self.lookback_returns = self.dual_momentum(self.monthly_prices, lookback, n_selection, long_only)
        elif signal_method == 'dual_momentum_sector_rotation':
            self.signal, self.abs_signal, self.rel_signal, self.lookback_returns = self.dual_momentum_sector_rotation(self.monthly_prices, lookback, n_selection, long_only)
        
        self.cost_df = self.transaction_cost(self.signal, cost)    
        self.portfolio_returns, self.monthly_returns, self.signal = self.backtest(signal_method, self.signal, self.monthly_returns, self.cost_df, start_date, end_date)
        self.performance, self.statistics, self.yearly_returns = self.performance_analysis(self.portfolio_returns, self.monthly_returns, monthly_data)
        
    
    def bool_converter(self, bool_var):
        """
        utility function
        """
        if bool_var == True:
            result = 1
        elif bool_var == False:
            result = 0

        return result 
    

    def get_returns(self, prices):
        """ 
        generate returns dataframe
        """
        returns = prices.pct_change().fillna(0)
        
        return returns


    def get_monthly_prices(self, prices):
        """
        generate monthly price dataframe (month end)
        """
        monthly_prices = pd.DataFrame()
        prices_subset = prices.copy()

        prices_subset["year"] = prices_subset.index.year
        prices_subset["month"] = prices_subset.index.month
        prices_subset["day"] = prices_subset.index.day
        month_end = prices_subset.groupby(["year","month"])["day"].max()

        for i in range(len(month_end)):
            day = "{}-{}-{}".format(month_end.index[i][0], month_end.index[i][1], month_end.iloc[i])
            monthly_prices = pd.concat([monthly_prices, prices_subset[prices_subset.index==day]])

        monthly_prices = monthly_prices.drop(['year', 'month', 'day'], axis=1)
        
        return monthly_prices


    def get_monthly_returns(self, monthly_prices):
        """
        generate monthly returns dataframe
        """
        monthly_returns = monthly_prices.pct_change().fillna(0)
        
        return monthly_returns


    def absolute_momentum(self, monthly_prices, lookback, long_only):
        """
        generate absolute momentum signal
        """
        # lookback_returns = monthly_prices.pct_change(periods=lookback).fillna(0)
        lookback_returns = (monthly_prices.shift(1) / monthly_prices.shift(lookback) - 1).fillna(0) # recent 1 month exclude
        long_signal = (lookback_returns > 0).apply(lambda col: col.map(self.bool_converter))        # long signal
        short_signal = -(lookback_returns < 0).apply(lambda col: col.map(self.bool_converter))      # short signal
        
        if long_only == True:   # long_only portfolio
            signal = long_signal 
        else:                   # long_short portfolio
            signal = long_signal + short_signal
        
        return signal, lookback_returns   


    def relative_momentum(self, monthly_prices, lookback, n_selection, long_only):
        """
        generate relative momentum signal
        """
        lookback_returns = (monthly_prices.shift(1) / monthly_prices.shift(lookback) - 1).fillna(0) # recent 1 month exclude
        rank = lookback_returns.rank(axis=1, ascending=False, method='max')
        long_signal = (rank <= n_selection).apply(lambda col: col.map(self.bool_converter))                                # long signal
        short_signal = -(rank >= len(rank.columns) - n_selection + 1).apply(lambda col: col.map(self.bool_converter))      # short signal
        
        if long_only == True:    # long_only portfolio
            signal = long_signal
        else:                    # long_short portfolio
            signal = long_signal + short_signal
            
        return signal, lookback_returns, rank


    def dual_momentum(self, monthly_prices, lookback, n_selection, long_only):
        """ 
        generate dual momentum signal
        """
        abs_signal, lookback_returns = self.absolute_momentum(monthly_prices, lookback, long_only)
        rel_signal, lookback_returns, rank = self.relative_momentum(monthly_prices, lookback, n_selection, long_only)
        signal = (abs_signal == rel_signal).apply(lambda col: col.map(self.bool_converter)) * abs_signal
        
        return signal, abs_signal, rel_signal, lookback_returns
    
    
    def dual_momentum_sector_rotation(self, monthly_prices, lookback, n_selection, long_only):
        """
        generate dual momentum sector rotation signal
        """
        snp500_prices = monthly_prices[['S&P 500']]
        monthly_prices_ex_sp500 = monthly_prices.drop(columns=['S&P 500'])
        abs_signal, lookback_returns = self.absolute_momentum(snp500_prices, lookback, long_only)
        rel_signal, lookback_returns, rank = self.relative_momentum(monthly_prices_ex_sp500, lookback, n_selection, long_only)
        
        abs_signal_broadcasted = pd.DataFrame({col: abs_signal.squeeze() for col in rel_signal.columns})
        signal = (abs_signal_broadcasted == rel_signal).apply(lambda col: col.map(self.bool_converter)) * abs_signal_broadcasted
        
        return signal, abs_signal, rel_signal, lookback_returns
    

    def transaction_cost(self, signal, cost):
        """
        calculate transaction cost
        """
        cost_df = (signal.diff() != 0).apply(lambda col: col.map(self.bool_converter)) * cost    # signal difference -> transaction occur -> costs charge
        cost_df.iloc[0] = 0                                                                      # The first line where a cost signal cannot emerge is treated as zero
        
        return cost_df


    def backtest(self, signal_method, signal, monthly_returns, cost_df, start_date, end_date):
        """
        backtesting with rebalancing data
        """
        signal = signal[start_date : end_date]
        monthly_returns = monthly_returns[start_date : end_date]
        cost_df = cost_df[start_date : end_date]
        
        if signal_method == 'absolute_momentum':
            portfolio_returns = signal * monthly_returns - cost_df   
            portfolio_returns.columns = [col + "_absolute_momentum" for col in portfolio_returns.columns]
            
        elif signal_method == 'relative_momentum':
            portfolio_returns = (signal * monthly_returns - cost_df).sum(axis=1)   
            portfolio_returns = pd.DataFrame(portfolio_returns, columns=['Relative_Momentum'])
            
        elif signal_method == 'dual_momentum': # 기초자산 둘 중에 하나에 대해 시그널 발생하므로 비중조절 필요없음
            portfolio_returns = (signal * monthly_returns - cost_df).sum(axis=1)   
            portfolio_returns = pd.DataFrame(portfolio_returns, columns=['Dual_Momentum'])
         
        elif signal_method == 'dual_momentum_sector_rotation':
            signal_count = signal.sum(axis=1).replace(0, np.nan)
            weights = signal.div(signal_count, axis=0)
            portfolio_returns = (weights  * monthly_returns - cost_df).sum(axis=1)   
            portfolio_returns = pd.DataFrame(portfolio_returns, columns=['Dual_Momentum_Sector_Rotation'])
        
        signal.plot(kind='area')

        return portfolio_returns, monthly_returns, signal
    
    
    def performance_analysis(self, portfolio_returns, monthly_returns, monthly_data):
        """ 
        performance, statistics, constituents weight
        """
        # performance
        portfolio_cumulative_returns = (np.cumprod(1 + portfolio_returns) -1).fillna(0)   
        bm_cumulative_returns = (np.cumprod(1 + monthly_returns) -1).fillna(0) 
        
        performance = portfolio_cumulative_returns.merge(bm_cumulative_returns, left_index=True, right_index=True)
        performance.plot()
        
        # statistics
        if monthly_data == True:
            
            statistics = pd.DataFrame()      

            # CAGR
            total_years = len(performance.index) / 12
            initial_value = 1000  
            final_value = initial_value * (1 + performance) 
            final_value = final_value.iloc[-1, :]
            cagr = (final_value / initial_value) ** (1 / total_years) - 1
            statistics['CAGR'] = cagr
            
            # Volatility
            vol_subset = portfolio_returns.merge(monthly_returns, left_index=True, right_index=True)
            vol_subset = np.log(1 + vol_subset)
            vol_subset = vol_subset.std() * np.sqrt(12)
            statistics['Volatility'] = vol_subset        
            
            # Sharpe
            sharpe = cagr / vol_subset
            statistics['Sharpe'] = sharpe   
            
            # Maximum DrawDowns
            cum = 1 + performance
            peak = cum.cummax()
            drawdown = (cum - peak) / peak
            statistics['MDD'] = drawdown.min()
            
            # Yearly Returns
            combined_returns = portfolio_returns.merge(monthly_returns, left_index=True, right_index=True)
            yearly_returns = (1 + combined_returns).resample('YE').prod() - 1
            yearly_returns.index = yearly_returns.index.year  # 연도만 추출
            
            statistics_formatted = statistics.copy()

            for col in statistics.columns:
                if col != 'Sharpe':
                    statistics_formatted[col] = statistics[col].apply(lambda x: f"{x * 100:.2f}%")
                else:
                    statistics_formatted[col] = statistics[col].apply(lambda x: f"{x:.4f}")

            yearly_returns_formatted = yearly_returns.applymap(lambda x: f"{x * 100:.2f}%")
            
            
            print("\n# 연도별 수익률 (포트폴리오 기준):")
            display(yearly_returns_formatted)
            
            print("")
            print(f"# 투자기간 : {total_years} 년")
            display(statistics_formatted)      
        
        return performance, statistics, yearly_returns


# %%
# Performance Decomposotion
start_date = '2010-01-01'
end_date = '2015-12-31'

equity_strategy = momentum_strategy(prices_equity, monthly_data=True, lookback=12, n_selection=1, signal_method='dual_momentum', cost=0.0007, long_only=True, start_date=start_date, end_date=end_date)
riskoff_strategy = momentum_strategy(prices_riskoff, monthly_data=True, lookback=36, n_selection=1, signal_method='dual_momentum', cost=0.0007, long_only=True, start_date=start_date, end_date=end_date)



# Composite Momentum
equity_dm = equity_strategy.portfolio_returns['Dual_Momentum']
riskoff_dm = riskoff_strategy.portfolio_returns['Dual_Momentum']

equity_dm.name = 'Equity_DualMomentum'
riskoff_dm.name = 'Riskoff_DualMomentum'

base_index = equity_dm.index
riskoff_aligned = riskoff_dm.reindex(base_index, method='nearest')

composite_dm = pd.concat([equity_dm, riskoff_aligned], axis=1)
composite_dm['Composite_DualMomentum'] = 0.5 * composite_dm['Equity_DualMomentum'] +  0.5 * composite_dm['Riskoff_DualMomentum']



# Performance Statistics
statistics = pd.DataFrame()      

# CAGR
total_years = len(composite_dm.index) / 12
initial_value = 100
cumulative_value = initial_value * (1 + composite_dm).cumprod()
final_value = cumulative_value.iloc[-1, :]
cagr = (final_value / initial_value) ** (1 / total_years) - 1
statistics['CAGR'] = cagr

# Volatility
vol_subset = composite_dm.copy()
vol_subset = np.log(1 + vol_subset)
vol_subset = vol_subset.std() * np.sqrt(12)
statistics['Volatility'] = vol_subset        

# Sharpe
sharpe = cagr / vol_subset
statistics['Sharpe'] = sharpe   

# Maximum DrawDowns
cum = (1 + composite_dm).cumprod()
peak = cum.cummax()
drawdown = (cum - peak) / peak
statistics['MDD'] = drawdown.min()

# Yearly Returns
combined_returns = composite_dm.copy()
yearly_returns = (1 + combined_returns).resample('YE').prod() - 1
yearly_returns.index = yearly_returns.index.year 


statistics_formatted = statistics.copy()

for col in statistics.columns:
    if col != 'Sharpe':
        statistics_formatted[col] = statistics[col].apply(lambda x: f"{x * 100:.2f}%")
    else:
        statistics_formatted[col] = statistics[col].apply(lambda x: f"{x:.4f}")

yearly_returns_formatted = yearly_returns.applymap(lambda x: f"{x * 100:.2f}%")

print("\n# 연도별 수익률 (포트폴리오 기준):")
print(yearly_returns_formatted)
yearly_returns.plot(kind='bar')

print("")
print(f"# 투자기간 : {total_years} 년")
print(statistics_formatted)  
cumulative_value.plot()





# %%

keyword = "S&P 500"
filtered_cols = [col for col in idx_daily.columns if keyword in col]

# 2. 해당 컬럼만 포함하는 데이터프레임 추출
filtered_df = idx_daily[filtered_cols]

target_index = [
    'S&P 500',
    'S&P 500 Consumer Discretionary',
    'S&P 500 Consumer Staples',
    'S&P 500 Energy',
    'S&P 500 Financials',
    'S&P 500 Health Care',
    'S&P 500 Industrials',
    'S&P 500 IT',
    'S&P 500 Materials',
    'S&P 500 Real Estate',
    'S&P 500 Communications',
    'S&P 500 Utilities'
    ]
    
    
prices_us_sector = idx_daily[target_index]
prices_us_sector.index = pd.to_datetime(prices_us_sector.index)
dmsr_strategy = momentum_strategy(prices_us_sector, monthly_data=True, lookback=12, n_selection=4, signal_method='dual_momentum_sector_rotation', cost=0.0007, long_only=True, start_date=start_date, end_date=end_date)
