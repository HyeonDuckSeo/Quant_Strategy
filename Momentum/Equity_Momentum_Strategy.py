# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xlwings as xw
import pickle
from IPython.display import display, HTML

pd.set_option('display.max_rows', 50)


# MSCI Indices
URL = 'C:/Users/tjgus/Desktop/msci_indices.xlsx'
book = xw.Book(URL)
prices = book.sheets(1).used_range.options(pd.DataFrame).value

# # USA Indices
# URL = 'C:/Users/shd4323/Desktop/usa_indices.xlsx'
# book = xw.Book(URL)
# prices = book.sheets(1).used_range.options(pd.DataFrame).value

display(prices)



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
            self.signal, self.abs_signal, self.rel_signal = self.dual_momentum(self.monthly_prices, lookback, n_selection, long_only)
        
        self.cost_df = self.transaction_cost(self.signal, cost)    
        self.portfolio_returns, self.monthly_returns = self.backtest(signal_method, self.signal, self.monthly_returns, self.cost_df, start_date, end_date)
        self.performance, self.statistics = self.performance_analysis(self.portfolio_returns, self.monthly_returns, monthly_data)
        
    
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
        lookback_returns = monthly_prices.pct_change(periods=lookback).fillna(0)
        long_signal = (lookback_returns > 0).applymap(self.bool_converter)   # long signal
        short_signal = -(lookback_returns < 0).applymap(self.bool_converter) # short signal
        
        if long_only == True:   # long_only portfolio
            signal = long_signal 
        else:                   # long_short portfolio
            signal = long_signal + short_signal
        
        return signal, lookback_returns   


    def relative_momentum(self, monthly_prices, lookback, n_selection, long_only):
        """
        generate relative momentum signal
        """
        lookback_returns = monthly_prices.pct_change(periods=lookback).fillna(0)
        rank = lookback_returns.rank(axis=1, ascending=False, method='max')
        long_signal = (rank <= n_selection).applymap(self.bool_converter)                               # long signal
        short_signal = -(rank >= len(rank.columns) - n_selection + 1).applymap(self.bool_converter)     # short signal
        
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
        signal = (abs_signal == rel_signal).applymap(self.bool_converter) * abs_signal
        
        return signal, abs_signal, rel_signal
        

    def transaction_cost(self, signal, cost):
        """
        calculate transaction cost
        """
        cost_df = (signal.diff() != 0).applymap(self.bool_converter) * cost    # signal difference -> transaction occur -> costs charge
        cost_df.iloc[0] = 0                                                    # The first line where a cost signal cannot emerge is treated as zero
        
        return cost_df


    def backtest(self, signal_method, signal, monthly_returns, cost_df, start_date, end_date):
        """
        backtesting with rebalancing data
        """
        signal = signal[start_date : end_date]
        monthly_returns = monthly_returns[start_date : end_date]
        cost_df = cost_df[start_date : end_date]
        
        if signal_method == 'absolute_momentum':
            portfolio_returns = signal.shift() * monthly_returns - cost_df.shift()   # prejudice bias prevention
            portfolio_returns.columns = [col + "_absolute_momentum" for col in portfolio_returns.columns]
            
        elif signal_method == 'relative_momentum':
            portfolio_returns = (signal.shift() * monthly_returns - cost_df.shift()).sum(axis=1)   # prejudice bias prevention
            portfolio_returns = pd.DataFrame(portfolio_returns, columns=['Relative_Momentum'])
            
        elif signal_method == 'dual_momentum':
            portfolio_returns = (signal.shift() * monthly_returns - cost_df.shift()).sum(axis=1)   # prejudice bias prevention
            portfolio_returns = pd.DataFrame(portfolio_returns, columns=['Dual_Momentum'])

        signal.shift().plot(kind='area', figsize=(20, 10))

        return portfolio_returns, monthly_returns
    
    
    def performance_analysis(self, portfolio_returns, monthly_returns, monthly_data):
        """ 
        performance, statistics, constituents weight
        """
        # performance
        portfolio_cumulative_returns = (np.cumprod(1 + portfolio_returns) -1).fillna(0)   
        bm_cumulative_returns = (np.cumprod(1 + monthly_returns) -1).fillna(0) 
        
        performance = portfolio_cumulative_returns.merge(bm_cumulative_returns, left_index=True, right_index=True)
        performance.plot(figsize=(20, 15))
        
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
            
            print(f"# 투자기간 : {total_years} 년")
            display(statistics)      
        
        return performance, statistics


# %%
# Performance Decomposotion
start_date = '1990-01-01'
end_date = '2023-12-31'

strategy = momentum_strategy(prices, monthly_data=True, lookback=12, n_selection=1, signal_method='dual_momentum', cost=0.0007, long_only=True, start_date=start_date, end_date=end_date)