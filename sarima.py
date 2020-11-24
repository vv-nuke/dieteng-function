import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.api import SARIMAX
import itertools
import scipy.stats as stats
from statsmodels.stats.diagnostic import acorr_ljungbox

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Sarima:

    def __init__(self, ts, last_timestamp=None):
        self.ts = ts
        self.last_timestamp = last_timestamp

    def initialize(self, order, seasonal_order):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = SARIMAX(self.ts, order=self.order, seasonal_order=self.seasonal_order,
                             enforce_stationarity=False, enforce_invertibility=False,
                             simple_differencing=False)
        self.results = self.model.fit(low_memory=True)

    def forecast(self, n=1) -> tuple:
        return self.results.forecast(n)

    def append(self, new_ts, last_timestamp=None, refit = False):
        if last_timestamp:
            self.last_timestamp = last_timestamp
        self.results = self.results.append(new_ts, refit=refit)

    def apply(self, new_init_ts, last_timestamp=None):
        self.init_ts = new_init_ts
        self.results = self.results.apply(new_init_ts)
        self.last_timestamp = last_timestamp


class SarimaAnalysis:

    def __init__(self, ts, period):
        self.ts = pd.Series(ts)
        self.period = period

    def s_diff(self):
        self.p_value = acorr_ljungbox(self.ts, lags=1)
        print('白噪声检验p值：', self.p_value[1], '\n')  # 大于0.05认为是白噪声，即序列在时间上不具有相关性

        self.ADF_value = adfuller(self.ts, autolag='BIC')  # p值为0小于0.05认为是平稳的(单位根检验)

        self.s_diff_ = self.ts.diff(self.period).dropna()  # 季节性差分

        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(211)  # 原始数据图
        ax1.plot(self.ts)
        ax2 = fig.add_subplot(212)  # 季节性查分差分后 无周期性 但是不平稳
        ax2.plot(self.s_diff_)
        plt.show()

    def acf_pacf_fig(self, lags=30):

        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(self.ts, lags=lags, ax=ax1)
        ax1.xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(self.ts, lags=lags, ax=ax2)
        ax2.xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        plt.show()


        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(self.s_diff_, lags=lags, ax=ax1)
        ax1.xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(self.s_diff_, lags=lags, ax=ax2)
        ax2.xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        plt.show()


    def order_select(self, order_sum_max=15):
        self.order_sum_max = order_sum_max
        # self.seasonal_order_sum_max = seasonal_order_sum_max
        seasonal_order = (0, 0, 0, 0, 0, 0)
        order_sum = sum(seasonal_order)
        self.bic_value = pd.DataFrame(columns=['seasonal_order', 'bic'])
        new_bic = (np.inf, None)
        bic = (np.inf, None)
        while order_sum < order_sum_max:
            order_waiting = []
            for i in range(6):
                _tmp = list(seasonal_order)
                _tmp[i] += 1
                order_waiting.append(_tmp)

            for _order in order_waiting:
                model = SARIMAX(self.ts,
                                order=_order[:3],
                                seasonal_order=_order[3:] + [self.period],
                                enforce_stationarity=False)
                results = model.fit(low_memory=True)
                bic_value_ = pd.DataFrame([_order, results.bic], columns=['seasonal_order', 'bic'])
                self.bic_value = pd.concat([self.bic_value, bic_value_])
                new_bic = min(new_bic, (results.bic, _order))
            if new_bic < bic:
                bic = new_bic
                new_bic = (np.inf, None)
                seasonal_order = tuple(bic[1])
                print(f"{bic[1]} is chosed with bic={bic[0]}")
            else:
                break
        self.order=seasonal_order
        print(f"best order is {self.order}")


    def sarima(self):
        model = SARIMAX(self.df.iloc[:, 1], order=self.param, seasonal_order=self.param_seasonal,
                        low_memory=True)  # 与上一句等价
        print('the best parameters: SARIMA{}x{}'.format(self.param, self.param_seasonal))
        self.results = model.fit()
        # joblib.dump(results, f'C:\\Users\\Administrator\\Desktop\\SARIMA模型.pkl')
        self.predict_ = self.results.forecast(self.forecast_num)

        fig, ax = plt.subplots(figsize=(20, 6))
        ax = self.predict_.plot(ax=ax)
        self.df.iloc[:, 1].plot(ax=ax)
        plt.legend(['y_pred', 'y_true'])
        plt.show()

    def sarima_(self):
        model = SARIMAX(self.df.iloc[:, 1], order=self.pdq_, seasonal_order=self.PDQ_)  # 与上一句等价
        print('the parameters: SARIMA{}x{}'.format(self.pdq_, self.PDQ_), '\n')
        self.results = model.fit()
        # joblib.dump(results, f'C:\\Users\\Administrator\\Desktop\\SARIMA模型.pkl')
        self.predict_ = self.results.forecast(self.forecast_num)

        fig, ax = plt.subplots(figsize=(20, 6))
        ax = self.predict_.plot(ax=ax)
        self.df.iloc[:, 1].plot(ax=ax)
        plt.legend(['y_pred', 'y_true'])
        plt.show()

    def model_eval(self):
        # 计算残差
        self.resid = self.results.resid

        # 模型检验
        # 残差的acf和pacf
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(311)
        fig = sm.graphics.tsa.plot_acf(self.resid.values.squeeze(), lags=40, ax=ax1)  # squeeze()数组变为1维
        ax2 = fig.add_subplot(312)
        fig = sm.graphics.tsa.plot_pacf(self.resid, lags=40, ax=ax2)
        # 残差自相关图断尾，所以残差序列为白噪声

    def qq_plot(self):
        plt.figure()
        stats.probplot(self.resid, dist="norm", plot=plt)
        plt.show()
        print('DW_value:', sm.stats.durbin_watson(self.resid))  # DW值接近于２时，说明残差不存在（一阶）自相关性