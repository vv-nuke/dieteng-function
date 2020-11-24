from statsmodels.tsa.arima.model import ARIMA
# import statsmodels.api as sm
# import scipy.stats as stats
# from statsmodels.stats.diagnostic import acorr_ljungbox
# from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
# import matplotlib.pylab as plt
import itertools
import time

import warnings

warnings.filterwarnings('ignore')


class Arima:

    def __init__(self, init_ts, last_timestamp=None):
        self.init_ts = init_ts
        self.last_timestamp = last_timestamp
        self.order = None
        self.model = None
        self.results = None

    def initialize(self, order, e_s=False):
        self.order = order
        self.model = ARIMA(self.init_ts, order=self.order, enforce_stationarity=e_s)
        self.results = self.model.fit()

    def forecast(self, n=1) -> tuple:
        return self.results.forecast(n)

    def append(self, new_ts, last_timestamp=None, refit=False):
        if last_timestamp:
            self.last_timestamp = last_timestamp
        self.results = self.results.append(new_ts, refit=refit)

    def apply(self, new_init_ts, last_timestamp=None):
        self.init_ts = new_init_ts
        self.results = self.results.apply(new_init_ts)
        self.last_timestamp = last_timestamp


class ArimaAnalysis:

    def __init__(self, ts):
        self.ts = ts

    # def diff_process(self):
    #     '''
    #     单位根检验按p值判断是否平稳，否则一直作差分直到序列平稳
    #     '''
    #     self.diff_ = pd.Series(self.ts)
    #     self.ADF = adfuller(self.diff_, autolag='BIC')
    #     self.d_ = 0
    #     while self.ADF[1] >= 0.01:
    #         self.diff_ = self.diff_.diff()  # 一次差分
    #         self.diff_ = self.diff_.dropna()
    #         self.ADF = adfuller(self.diff_, autolag='BIC')
    #         # 1%、%5、%10不同程度拒绝原假设的统计值和ADF Test result的比较,
    #         # ADF Test result同时小于1%、5%、10%说明非常好的拒绝原假设，p值小于0.05，则平稳
    #         # print('ADF检验:', '\n', self.ADF_value, '\n')
    #         self.d_ += 1
    #     self.ljungbox = acorr_ljungbox(self.ts, lags=1)
    #
    #     print(f'{self.d_} 次差分后，序列平稳')
    #     print(f'ADF 检验p值(>0.01)：{self.ADF[1]}，白噪声检验p值(<0.05)：{self.ljungbox[1]}\n')  # 大于0.05认为是白噪声，即序列在时间上不具有相关性
    #     # self.ADF_value = ADF(self.df.iloc[:,0]) #p值为0小于0.05认为是平稳的(单位根检验)
    #
    #     fig = plt.figure(figsize=(20, 6))
    #     ax1 = fig.add_subplot(211)  # 原始数据图
    #     ax1.plot(self.ts)
    #     ax2 = fig.add_subplot(212)  # 再一次差分之后 平稳
    #     ax2.plot(self.diff_)
    #     plt.show()

    # def acf_pacf_fig(self, lags=30):
    #
    #     fig = plt.figure(figsize=(12, 8))
    #     ax1 = fig.add_subplot(211)
    #     fig = sm.graphics.tsa.plot_acf(self.ts, lags=lags, ax=ax1)
    #     ax1.xaxis.set_ticks_position('bottom')
    #     fig.tight_layout()
    #     ax2 = fig.add_subplot(212)
    #     fig = sm.graphics.tsa.plot_pacf(self.ts, lags=lags, ax=ax2)
    #     ax2.xaxis.set_ticks_position('bottom')
    #     fig.tight_layout()
    #     plt.show()
    #
    #     fig = plt.figure(figsize=(12, 8))
    #     ax1 = fig.add_subplot(211)
    #     fig = sm.graphics.tsa.plot_acf(self.diff_, lags=lags, ax=ax1)
    #     ax1.xaxis.set_ticks_position('bottom')
    #     fig.tight_layout()
    #     ax2 = fig.add_subplot(212)
    #     fig = sm.graphics.tsa.plot_pacf(self.diff_, lags=lags, ax=ax2)
    #     ax2.xaxis.set_ticks_position('bottom')
    #     fig.tight_layout()
    #     plt.show()

    def order_select(self, pdq_range=(range(3), range(3), range(3))):
        p_range, d_range, q_range = pdq_range[:]
        pdq_list = list(itertools.product(p_range, d_range, q_range))
        self.bic_value = pd.DataFrame(columns=['order', 'bic'])
        bic = (np.inf, None)
        start_params = None
        for _order in pdq_list:
            _s = time.time()
            model = ARIMA(self.ts, order=_order, enforce_stationarity=False)
            results = model.fit(start_params=start_params, low_memory=True)
            start_parmas = results.params
            bic_value_ = pd.DataFrame([[_order, results.bic]],
                                      columns=['order', 'bic'])
            self.bic_value = pd.concat([self.bic_value, bic_value_])
            # print(f"{_order} take {round(time.time() - _s, 2)}s")
            bic = min(bic, (results.bic, _order))
        self.best_order = bic
        print(f"best order is {self.best_order}")
        return self.best_order
        # (self, pdq_max=(3, 2, 3)):
        # self.order = st.arma_order_select_ic(self.diff_, max_ar=self.p_max, max_ma=self.q_max,
        #                                      ic=['aic', 'bic', 'hqic'])
        # '''
        # 常用的是AIC准则，AIC鼓励数据拟合的优良性但是尽量避免出现过度拟合(Overfitting)的情况。所以优先考虑的模型应是AIC值最小的那一个模型。
        # 为了控制计算量，限制AR最大阶和MA最大阶，但是这样带来的坏处是可能为局部最优
        # order.bic_min_order返回以BIC准则确定的阶数，是一个tuple类型
        # '''
        # self.order.bic_min_order = list(self.order.bic_min_order)
        # self.order.bic_min_order.insert(1, self.i)
        # self.order.bic_min_order = tuple(self.order.bic_min_order)
        # print('the best parameters: ARIMA{}'.format(self.order.bic_min_order))

    # def arima(self):
    #     model = ARIMA(self.data.iloc[:, 1], order=self.order.bic_min_order)
    #     self.results = model.fit()
    #     # joblib.dump(results, f'C:\\Users\\Administrator\\Desktop\\ARIMA模型.pkl')
    #     self.predict = self.results.forecast(self.forecast_num)
    #     self.predict = self.predict[0]  # 预测值还包含一个区间，这里只取预测的一个值
    #     # ------------------------------------人工修正-----------------------------------------#
    #     self.predict_ = []
    #     self.correct_value = self.correct_value.values
    #     for i in range(len(self.data), len(self.correct_value)):
    #         self.predict_.append(self.predict[i - len(self.data)] + self.correct_value[i][0])
    #     # ------------------------------------------------------------------------------------#
    #     fig, ax = plt.subplots(figsize=(30, 6))
    #     self.predict_and_df = np.concatenate((np.array(self.data.iloc[:, 1]), self.predict_))
    #     dt1 = {'x': [], 'y': []}
    #     for i in range(len(self.predict_and_df)):
    #         dt1['x'].append(i)
    #         dt1['y'].append(self.predict_and_df[i])
    #
    #     self.plot_dt1 = pd.DataFrame(dt1, columns=['x', 'y'])
    #     plt.plot(self.plot_dt1.x[:len(self.data.iloc[:, 1])], self.plot_dt1.y[:len(self.data.iloc[:, 1])])
    #     plt.plot(self.plot_dt1.x[len(self.data.iloc[:, 1]) - 1:], self.plot_dt1.y[len(self.data.iloc[:, 1]) - 1:])
    #     # plt.plot(self.plot_dt2.x[len(self.df.iloc[:, 1])-1:], self.plot_dt2.y[len(self.df.iloc[:, 1])-1:], color = 'blue')
    #     plt.legend(['y_true', 'y_pred'])
    #     # plt.savefig('C:\\Users\\Administrator\\Desktop\\TimeSeries\\300个预测20个\\' + 'ARIMA{}'.format(self.order.bic_min_order) + f'{self.ii}')
    #     plt.show()
    #
    # def model_eval(self):
    #     # 计算残差
    #     self.resid = self.results.resid
    #
    #     # 模型检验
    #     # 残差的acf和pacf
    #     fig = plt.figure(figsize=(12, 8))
    #     ax1 = fig.add_subplot(311)
    #     fig = sm.graphics.tsa.plot_acf(self.resid.values.squeeze(), lags=40, ax=ax1)  # squeeze()数组变为1维
    #     ax2 = fig.add_subplot(312)
    #     fig = sm.graphics.tsa.plot_pacf(self.resid, lags=40, ax=ax2)
    #     # 残差自相关图断尾，所以残差序列为白噪声
    #
    # def qq_plot(self):
    #     plt.figure()
    #     stats.probplot(self.resid, dist="norm", plot=plt)
    #     plt.show()
    #     print('DW_value:', sm.stats.durbin_watson(self.resid), '*' * 70)  # DW值接近于２时，说明残差不存在（一阶）自相关性
