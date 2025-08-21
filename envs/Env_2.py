import pandas as pd 
import numpy as np

import gym
from gym import spaces

'''
Pnadas & numpy : handle my financial time-series data and numerical operations.
gym and spaces: From OpenAI Gym, used to build reinforcement learning environment.
    - gym.Env: is the base class my environment inherits form.
    - spaces: define the action space (what the agent can do) and observation space (what the agent sees). 
'''

class Mult_asset_env(gym.Env):
    '''Defines my custom multi-asset trading environment, following Gym's API(reste, step, render).
    The reason we have this class is my simulation world for reinforcement learning. A place that agent
    can see the market data snapshot and makes a descions buy/sell/hold. We have in separate file and not next
    to rest of codes due to:
        - separation of concerns and keep it clean in that case agent can only focuse on simulate the market correctly
        - reusability: since we test different algorithms (A2C, PPO) this environment was convenience for both of them
        and there was no need to rewrite seperate envrionment for each of them'''
    
    def __init__(
        self,
        df, 
        num_assets, 
        features_list, 
        window_size=5, 
        initial_balance=1000, 
        transaction_cost_rate=0.001,
        risk_aversion=0.01,
        cash_penalty=0.1,
        invest_bonus=0.1,
        vol_window=20,
        clip_return=0.10,
        verbose = False
    ):
        ''' initialize the market world'''
        
        super().__init__()
        
        if 'Date' not in df.columns:
            df = df.reset_index().rename(columns = {df.index.name or 'index': 'Date'})
        
        self.df = df.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.tickers = sorted(self.df['Ticker'].unique())
        self.num_assets = num_assets
        self.features_list = features_list # like (close price, volume, RSI,...)
        self.num_features_per_asset = len(features_list)
        self.window_size = window_size # how many past timesteps the agent sees
        self.initial_balance = initial_balance # cash + balance
        self.transaction_cost_rate = transaction_cost_rate
        self.verbose = verbose
        self.risk_aversion = float(risk_aversion) # reduces reward if portfolio is too risky.
        self.cash_penalty_coef = float (cash_penalty) # penalizes keeping too much idle cash.
        self.invest_bonus_coef = float (invest_bonus) # optional reward for actually deploying capital.
        self.vol_window = int(vol_window)
        self.clip_return = float(clip_return)
        self.trade_history = []
        
        self.action_space = spaces.Box(low = 0.0,
                                       high = 1.0,
                                       shape= (self.num_assets + 1,),
                                       dtype=np.float32)
        
        '''
        - action space: agent outputs a vevtor like [0.2, 0.5] which means 20% in asset 1, 50% in asset 2
        - contains the rolling window of features (price, indicators) for all assets
        '''
        
        num_portfolio_features = self.num_assets + 1
        observation_row_size = self.num_assets * self.num_features_per_asset + num_portfolio_features
        
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(window_size, 
                                            observation_row_size),
                                            dtype=np.float32)
        
        self.dates = self.df['Date'].unique()
        self.current_step = self.window_size # keeps track of where we are in the dataset.
        
        self._daily_idx = {}
        for d in self.dates:
            block = self.df[self.df['Date'] == d].sort_values('Ticker')
            if len(block) != self.num_assets:
                block = block.set_index('Ticker').reindex(self.tickers).reset_index()
            self._daily_idx[d] = block
        
        self.reset()

        '''
        why __init__ is important?
            - we have a market world ready to simulate.
            - the agent sees states in a fixed structure.
            - the agent's actions are bounded and realistic.
            - financail constraints are enforced.
        '''
    
    def reset(self):
        ''' The goal of this function is to set the environment to the initial state
            for a new episode.'''
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.cash = self.initial_balance
        self.portfolio_weights = np.array([0.0] * self.num_assets + [1.0],dtype=np.float32)
        self.assets_shares = np.array([0.0] * self.num_assets, dtype=np.float32)
        self.avg_cost_per_asset = np.zeros(self.num_assets, dtype=np.float32)
        self.history = []
        self.return_history = []
        self.trade_history = []
        self.peak_portfolio_value = self.initial_balance
        return self.get_obs()
    
    def _window_df(self, start_idx, end_idx):
        ''' The goal is return a subset of the full dataframe containing only the rows between start_idx and end_idx.
        this is my agent's window of observations.'''

        start_date = self.dates[start_idx]
        end_date = self.dates[end_idx]
        obs_df = self.df[(self.df['Date'] >= start_date) & (self.df['Date'] < end_date)]
        
        return obs_df
            
    def get_obs(self):
        '''This function converts a slice of the historical price/features data into
        a normalized observation array suitable for RL.'''

        start_idx = self.current_step - self.window_size
        end_idx = self.current_step      
        
        obs_df = self._window_df(start_idx, end_idx)
        '''
        use window_df to get the relevant historical data for the agent's current step
        '''
        expected_rows = self.window_size * self.num_assets
        obs_values = obs_df[self.features_list].values.astype(np.float32)
        
        if obs_values.shape[0] < expected_rows:
            pad_rows = expected_rows - obs_values.shape[0]
            obs_values = np.vstack([obs_values, np.zeros((pad_rows, len(self.features_list)), dtype=np.float32)])
        '''
        at the very beginning, there might not be enough rows to fill the window. So,
        we pad with zeros to ensure the observation has consistent shape.
        '''
        mean = np.mean(obs_values, axis=0, keepdims=True)
        std = np.std(obs_values, axis=0, keepdims=True) + 1e-8
        obs_values = (obs_values - mean) / std
        '''
        Standardizes each feature to have zero mean and unit varance.
        and avoids numerical instability with 1e08 add to std.
        '''

        feature_window_data = obs_values.reshape(
            self.window_size, self.num_assets * self.num_features_per_asset
        )
        '''
        combines all assets into a single feature vector per timestep.
        '''
        
        current_portfolio_state_repeated = np.tile(self.portfolio_weights, (self.window_size, 1))
        obs = np.concatenate([feature_window_data, current_portfolio_state_repeated], axis=1)
        '''
        Repeats current portfolio weights across the window to give the agent context.
        '''
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        '''
        ensures the observation is always numeric, avoiding NaNs or infinite values that
        break training.
        '''
        return obs.astype(np.float32)
    
    def step(self, action):
        '''It simulates one trading step/day in your portfolio environment.'''

        if self.current_step >= len(self.dates):
            return self.get_obs(), 0.0, True, {'reason': 'end_of_data'}
        ''' The goal is stop the episode if we have reached the end of the available data.'''
        
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        s = float(np.sum(action))
        if s <= 1e-12:
            action = np.ones_like(action, dtype=np.float32) / (self.num_assets + 1)
        else:
            action = action / s
        '''
        Normalize actions:
            - converts the action to a flat float32 array.
            - ensures that the sum of allocations is 1 (i.e., proportions of portfolio).
            - if the sum is evtremely small (practically zero), use an equal weight allocation
            including cash.
        '''
        
        self.portfolio_weights = action

        current_date = self.dates[self.current_step]
        block = self._daily_idx[current_date]
        current_close_prices = block['Close'].values.astype(np.float32) 
        current_close_prices = np.nan_to_num(current_close_prices, nan=0.0)
        current_close_prices = np.maximum(current_close_prices, 1e-6)
        '''
        Extract current market prices:
            - retrieves the closing prices for all assets on the current step.
            - cleans the data
                - replace NaNs with 0.
                - avoid prices being exactly zero (minimum 1e-6)
        '''
        
        current_assets_mv = np.sum(self.assets_shares * current_close_prices)
        pv_before = float(self.cash + current_assets_mv)
        '''
        compute current portfolio value:
            - calculates market value of assets (current_assets_mv)
            - Total portfolio value pv_before = cash + assets value.
        '''
        if pv_before <= 0:
            obs = self.get_obs()
            return obs, -100, True, {'reason': 'bankrupt'}
        ''' if portfolio value is 0 or negative -> end episode with a large negative reward.'''
        
        target_assets_value = action[:-1] * pv_before
        target_assets_shares = target_assets_value / current_close_prices
        shares_to_buy_sell = target_assets_shares - self.assets_shares
        '''
        Compute traget holdings:
            - valvulate desired value per asset according to action weights.
            - convert desired value -> desired number of shares.
            - determine buy/sell qunatity for each asset.
        '''
        
        transaction_costs = np.sum (np.abs(shares_to_buy_sell * current_close_prices)) * self.transaction_cost_rate
        new_cash = self.cash - np.sum(np.maximum(shares_to_buy_sell,0)* current_close_prices) \
            + np.sum(np.abs(np.minimum(shares_to_buy_sell,0) * current_close_prices)) \
            - transaction_costs
        '''
        Transaction costs and cash update:
            - Transaction costs = sum of obsolute trades * transaction rate.
            - cash after trades:
                - subtract cost for buying shares.
                - add value from selling shares.
                - subtract transactions costs.
        '''
        
        self.assets_shares += shares_to_buy_sell
        self.cash = new_cash

        if new_cash < 0:
            obs = self.get_obs()
            return obs, -100.0, True, {'reason': 'infeasible_trade'}
        '''
        Updates asset holdings and check feasibility:
            - update holdings and cash
            - if cash<0 -> infeasible trade -> episode ends with penalty.
        '''
        
        for i in range(self.num_assets):
            if shares_to_buy_sell[i] > 0:
                old_shares = self.assets_shares[i] - shares_to_buy_sell[i]
                total_cost_existing = self.avg_cost_per_asset[i] * old_shares
                total_cost_new = shares_to_buy_sell[i] * current_close_prices[i]
                total_shares = old_shares + shares_to_buy_sell[i]
                if total_shares > 0:
                    self.avg_cost_per_asset[i] = (total_cost_existing + total_cost_new) / total_shares
        '''
        update average cost per asset:
            - Tracks average purchase cost per asset for later calculations(e.g., P&L, risk management).
        '''
        self.last_transaction_costs = transaction_costs
        self.cash = new_cash
        """ Update cash"""
        
        new_assets_mv = float(np.sum(self.assets_shares * current_close_prices))
        self.balance = float(self.cash + new_assets_mv)
        """ Update portfolio value"""

        step_log = {
            'Step': self.current_step,
            'Date': str(current_date),
            'Cash': round(self.cash, 2),
            'PortfolioValue': round(self.balance, 2),
        }
        
        for i, ticker in enumerate(self.tickers):
            step_log[f'{ticker}_Shares'] = round(float(self.assets_shares[i]), 4)
            step_log[f'{ticker}_Price'] = round(float(current_close_prices[i]), 2)
            step_log[f'{ticker}_Value'] = round(float(self.assets_shares[i] * current_close_prices[i]), 2)

        self.trade_history.append(step_log)
        '''
        - logs current step details : cash, portfolio value, asset holdings, prices.
        - helps analze trades later.
        '''

        daily_ret = (self.balance - pv_before) / (pv_before + 1e-9)
        daily_ret = float(np.clip(daily_ret, -self.clip_return, self.clip_return))
        self.return_history.append(daily_ret)
        
        w = min(len(self.return_history), self.vol_window)
        vol = float(np.std(self.return_history[-w:])) if w >= 2 else 0.0
        
        w_cash = float(self.portfolio_weights[-1])
        invested_weight = 1.0 - w_cash
        
        profit_bonus = 0.0

        if np.any(self.assets_shares > 0) and len(self.return_history) > 0:
            recent_returns = self.return_history[-5:] if len(self.return_history) >= 5 else self.return_history
            positive_returns = np.mean(np.maximum(recent_returns, 0))
            profit_bonus = positive_returns * 0.1
        
        reward = (
            daily_ret
            - self.risk_aversion * vol
            - self.cash_penalty_coef  * w_cash
            + self.invest_bonus_coef * invested_weight
            + profit_bonus
        )
        '''
        compute reward:
            - reward is mostly risk-adusted return
                - daily return 
                - penalized by volatlity and cash weight
                - bonus for invested portion and recent positive returns.
        '''

        self.history.append(self.balance)
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.balance)
        '''
        update history and peak portfolio value:
            - track portfolio value evolution.
            - track the peak value for drawdown caluations.
        '''
        
        
        done = (self.current_step + 1) >= len(self.dates) or self.balance < (self.initial_balance * 0.1)
        ''' 
        check if episode is done:
            - reached last date.
            - portfolio value drops below 10% of initial
        '''
        if done and (self.current_step + 1) >= len(self.dates):
            final_prices = current_close_prices
            self.balance = self.cash + np.sum(self.assets_shares * final_prices)
        else:      
            if not done:
                self.current_step += 1
        '''' Move to next time step if the episode is not over'''

        info = {}
        if done:
            if self.balance <= 0: info['reason'] = 'bankrupt'
            elif self.balance < (self.initial_balance * 0.1): info['reason'] = 'low_balance'
            elif self.current_step >= len(self.dates): info['reason'] = 'end_of_data'
        '''
        debug section: provides extra info about why episode ended.
        '''
        obs = self.get_obs()

        if self.verbose and (self.current_step % 50 ==0):
             print(f"[env step {self.current_step}] balance={self.balance:.2f}, cash={self.cash:.2f}, shares={self.assets_shares}")
        ''' solving feezing problem: print debug info every 50 setps if verbose mode is enabled.'''
        return obs, float(reward), bool(done), info

    def render(self, mode='human'):
        ''' print to console'''
        if mode == 'human':
            holdings_str = ', '.join([
                f'{ticker}: {shares:.2} shares'
                for ticker, shares in zip(self.tickers, self.assets_shares)
            ])
            print(
                f'Step {self.current_step} | '
                f'PV: {self.balance:.2f} | '
                f'Cash: {self.cash:.2f} | '
                f'TxCost: {self.last_transaction_costs:.4f} | '
                f'Weights(asset...,cash): {np.round(self.portfolio_weights, 3).tolist()} | '
                f'Holdings: {holdings_str}'
            )
    