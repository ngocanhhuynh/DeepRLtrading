from __future__ import print_function

import datetime
import numpy as np
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web

import pylab as plt
from matplotlib import cm
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix, accuracy_score
from itertools import permutations
import warnings
warnings.filterwarnings("ignore")

class Ticker:

    def __init__(self, symbol = "vnm", fr_date = "2017-11-01", to_date = "2018-08-01"):
        self.symbol = symbol
        self.fr_date = fr_date   
        self.to_date = to_date
        self.nr_hid_states = 3

        self.hidden_states, self.quotes = self._hmm()
        
    def _discretise(self, x):
        out = np.ones(x.shape)
        nr_of_points = 50
        start = np.min(x, axis=0)
        stop = np.max(x, axis=0) 
        if len(x.shape)==1:
            bins_array = np.linspace(start, stop, nr_of_points)
            out = np.digitize(x, bins_array)
        elif len(x.shape)>1:
            for i in range(x.shape[1]):
                bins_array = np.linspace(start[i], stop[i], nr_of_points)
                out[:,i] = np.digitize(x[:,i], bins_array)  
        return out
    
    def _crawl(self, fr_date, to_date, VN = True):                             #'yyyy-mm-dd'  #(year, month, day)
        if not VN:
            # get quotes from yahoo finance
            quotes = web.DataReader(self.symbol, "yahoo", datetime.date(*fr_date), datetime.date(*to_date))
        
        else:
            link = 'https://raw.githubusercontent.com/88d52bdba0366127fffca9dfa93895/vnstock-data/master/symbols/'
            df = pd.read_csv(link + self.symbol + '.csv')
            df = df[(df['date'] >= fr_date) & (df['date'] <= to_date)]
            df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volumn': 'Volumn'})
            df = df.set_index('date')
            quotes = df.sort_index(ascending=True)
            self.quotes = quotes
            
        self.dates = self.quotes.index.values
        self.quotes['logPrice'] = self.quotes['Close'].map(lambda x: np.log(x))
        self.quotes['logReturn']= self.quotes['logPrice'].diff(1)
        self.quotes['logReturn_idx']= self._discretise(self.quotes['logReturn'])

        self.fracChange = (self.quotes['Close'] - self.quotes['Open'])/self.quotes['Open']
        self.fracHigh = (self.quotes['High'] - self.quotes['Open'])/self.quotes['Open']
        self.fracLow = (self.quotes['Open'] - self.quotes['Low'])/self.quotes['Open']
        hmm_inp = self._discretise(np.column_stack([self.fracChange, self.fracHigh, self.fracLow])) 

        return hmm_inp, quotes

    def _hmm(self):
        """
        =========================================================
        Hidden Markov Model with Gaussian emissions on stock data
        =========================================================
        """
        np.random.seed(0)

        n_components=self.nr_hid_states
        n_mix=4
        min_covar=0.001
        startprob_prior=1.0
        transmat_prior=1.0
        weights_prior=1.0
        means_prior=0.0
        means_weight=0.0
        covars_prior=None
        covars_weight=None
        algorithm='viterbi'
        covariance_type='diag'
        random_state=None
        n_iter=15
        tol=0.01
        verbose=False
        params='stmcw'
        init_params='stmcw'

        #model = hmm.GMMHMM(n_components=n_components, n_mix=n_mix, n_iter=n_iter, verbose=verbose)       
        model = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter, verbose=verbose)  
        self.model = model

        # Crawl training data
        hmm_train, _= self._crawl(fr_date="2014-11-01", to_date="2017-11-01")
        model.fit(hmm_train)

        # Crawl testing data
        hmm_test, quotes = self._crawl(fr_date=self.fr_date, to_date=self.to_date)
        hidden_states = model.predict(hmm_test)

        return hidden_states, quotes
      
    def print_para(self):
        
        print("Transition Matrix")
        print(self.model.transmat_)
        print()

        print("Means and Vars of each hidden state")
        
        for i in range(self.model.n_components):
            print("hidden state #%d" % i)
            print("mean = ", self.model.means_[i])
            print("var = ", np.diag(self.model.covars_[i]))
            print()
      
    def plot_all_states(self):
        years = YearLocator()   # every year
        months = MonthLocator()  # every month
        yearsFmt = DateFormatter('%Y')
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)

        ax.plot(self.dates, self.fracChange, 'g-', alpha=.4)

        for i in range(self.model.n_components):
            # use fancy indexing to plot data in each state
            idx = (self.hidden_states == i)
            ax.plot_date(self.dates[idx], self.fracChange[idx], 'o', label="hidden state $\# %d$" % i)
    
        ax.legend()

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        ax.autoscale_view()

        # format the coords message box
        ax.fmt_xdata = DateFormatter('%Y-%m-%d')
        ax.fmt_ydata = lambda x: '$%1.2f' % x
        ax.grid(True)

        fig.autofmt_xdate()
        plt.title('fracChange by GHMM Hidden States predicted on stock %s\
\n from %s to %s' % (self.symbol.upper(), self.fr_date, self.to_date))
        plt.show()
        
    def plot_by_state(self):
        from matplotlib import cm

        modelfig, axs = plt.subplots(self.model.n_components, sharex=True, sharey=True, figsize=(20,10))
        colours = cm.rainbow(np.linspace(0, 1, self.model.n_components))

        for i, (ax, colour) in enumerate(zip(axs, colours)):
            ax.plot(self.dates, self.fracChange, 'g-', alpha=.4)
            # Use fancy indexing to plot data in each state.
            idx = self.hidden_states == i
            ax.plot_date(self.dates[idx], self.fracChange[idx], "o", c=colour)              
            ax.set_title("{0}th hidden state".format(i))
            # Format the ticks.
            ax.xaxis.set_major_locator(YearLocator())
            ax.xaxis.set_minor_locator(MonthLocator())
            ax.grid(True)
        # plt.title('FracChange by GHMM Hidden States predicted on stock %s\
# \n from %s to %s' % (self.symbol.upper(), self.dates[0], self.dates[-1]))
        plt.show()
        
    def plot_hist(self):
        #fracChange :=  ('Close' - 'Open') / 'Open'
        plt.hist(self.fracChange[self.hidden_states==0], alpha=.6, label='hidden_states==0')
        plt.hist(self.fracChange[self.hidden_states==1], alpha=.6, label='hidden_states==1')
        plt.hist(self.fracChange[self.hidden_states==2], alpha=.6, label='hidden_states==2')
        plt.legend()
        # plt.title('%s $fracChange$ from %s to %s' % (self.symbol, self.fr_date, self.to_date))
        plt.title('Distribution of GHMM Hidden States predicted on stock %s\
\n from %s to %s' % (self.symbol.upper(), self.dates[0], self.dates[-1]))
        plt.show()
        
    def metrics(self, neutral_band_lower=0, neutral_band_upper=0, labels=[0,1,2]):     #[0,1,2]:= [bullish, bearish, neutral]    
        self.expected_states = self.fracChange.map(lambda x: labels[0] if x>neutral_band_upper 
                                                          else (labels[1] if x<neutral_band_lower else labels[2])) 
        
        conf_mat = confusion_matrix(self.expected_states, self.hidden_states, labels=labels)
        acc_score = accuracy_score(self.expected_states, self.hidden_states)
        
        return conf_mat, acc_score

def main():
	vn30port = {"bmp",	"cii", "ctd", "ctg", "dhg",	"dpm", "fpt", "gas", "gmd",	"hpg", 
				"hsg", "kdc", "mbb", "msn", "mwg", "nvl", "plx", "pnj",	"ree", "ros", 
				"sab", "sbt", "ssi", "stb", "vcb", "vic", "vjc", "vnm", "vpb", "vre"}

	# except_tickers = {"ctd", "pnj", "ros", "sab", "vpb", "vre"}
    except_tickers = {" "}
	
	tickers_list = sorted(list(vn30port - except_tickers))
	for symbol in tickers_list:
		ticker = Ticker(symbol = symbol)

		labels_list = list(permutations([0,1,2]))
		#[(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
		#[labels[0], labels[1], labels[2]:= [bullish, bearish, neutral] 
		temp1 = 0
		temp2 = []		
		for labels in labels_list:
			conf_mat, acc_score = ticker.metrics(neutral_band_lower = -0.01, neutral_band_upper = 0.01, labels = labels)
			if acc_score >= temp1: 
				temp1 = acc_score
				temp2 = labels
			else: continue
			#print("Confusion Matrix: \n", conf_mat)
			#print()
		print('SYMBOL:{0} \ ACCURACY:{1:0.6f} \ LABELS:{2}'.format(
			symbol.upper(), temp1, temp2))
		print()

		ticker.plot_hist()
		ticker.print_para()
		ticker.plot_all_states()
		ticker.plot_by_state()


if __name__ == '__main__':
    main()
