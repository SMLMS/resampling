#!/uqpropertysr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:57:32 2019

@author: malkusch
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import curve_fit

class Hill:
    def __init__(self):
        self._data = []
        self._fitSuccess = bool(),
        self._n = int()
        self._ka = float()
        self._initKa = float()
        self._kd = float()
        self._initKd = float()
        self._thetaLgnd = float()
        self._thetaCmplx = float()
        self._initThetaCmplx = float()
        self._logL = float()
        self._bic = float()
        self._aic = float()
        self._aicc = float()
        self._rmse = float()
        self.fitFailed()
        
    @property
    def data(self):
        return np.copy(self._data)
    
    @data.setter
    def data(self, values: np.ndarray):
        self._data = np.copy(values)
    
    @property
    def fitSuccess(self):
        return self._fitSuccess
        
    @property
    def n(self):
        return self._n
    
    @n.setter
    def n(self, value: int):
        self._n = value

    @property
    def ka(self):
        return self._ka
    
    @property
    def initKa(self):
        return self._initKa
    
    @property
    def kd(self):
        return self._kd
    
    @property
    def initKd(self):
        return self._initKd
    
    @initKd.setter
    def initKd(self, value: float):
        self._initKd = value
        self._initKa = self.initKd ** (1.0/float(self.n))
        
    @property
    def thetaLgnd(self):
        return self._thetaLgnd
    
    @thetaLgnd.setter
    def thetaLgnd(self, value: float):
        self._thetaLgnd = value
    
    @property
    def thetaCmplx(self):
        return self._thetaCmplx
    
    @property
    def initThetaCmplx(self):
        return self._initThetaCmplx
    
    @initThetaCmplx.setter
    def initThetaCmplx(self, value: float):
        self._initThetaCmplx = value
    
    @property
    def logL(self):
        return self._logL
    
    @property
    def bic(self):
        return self._bic

    @property
    def aic(self):
        return self._aic
    
    @property
    def aicc(self):
        return self._aicc
    
    @property
    def rmse(self):
        return self._rmse

    def hillEquation(self, c, t, k):
        theta = self.thetaLgnd + ((t - self.thetaLgnd) * (c**self.n) / (k**self.n + c**self.n))
        return theta
    
    def negLogLikelihood(self, para, cData, thetaData):
        thetaHat = self.hillEquation(cData, para[0], para[1])
        #std = np.std(thetaHat - thetaData)
        std = 0.1
        ll = np.sum(norm.logpdf(x = thetaData, loc = thetaHat, scale = std))
        return -1.0 * ll
    
    def fitMLE(self):
        fitRslt = minimize(fun = self.negLogLikelihood,
                           x0 = [self.initThetaCmplx, self.initKa],
                           args = (self._data[:,0], self._data[:,1]),
                           method = "Nelder-Mead",
                           #bounds = (-np.inf, np.inf),
                           options = {'disp': False})
        if(fitRslt.success):
            self._fitSuccess = True
            self._thetaCmplx = fitRslt.x[0]
            self._ka = fitRslt.x[1]
            self._kd = self.ka ** self.n
            self.evaluateModel()
        else:
           self.fitFailed() 
        return fitRslt
    
    def fitLSQ(self):
        fitRslt, fitCov = curve_fit(f = self.hillEquation,
                                    xdata = self._data[:,0],
                                    ydata = self._data[:,1], 
                                    p0 = [self.initThetaCmplx, self.initKa],
                                    #bounds = (-np.inf, np.inf),
                                    method = 'trf')
        self._fitSuccess = True
        self._thetaCmplx = fitRslt[0]
        self._ka = fitRslt[1]
        self._kd = self.ka ** self.n
        self.evaluateModel()
        return fitRslt
    
    def evaluateModel(self):
        para = 2
        obs = np.shape(self._data)[0]
        self._logL = - self.negLogLikelihood([self.thetaCmplx, self.ka], self._data[:,0], self._data[:,1])
        self._bic = (-2.0*self.logL) + (para * np.log(obs))
        self._aic = 2.0 * para - 2.0 * self.logL
        self._aicc = self.aic + (((2.0 * para**2) + (2.0 * para))/(obs - para - 1.0))
        self._rmse = np.sqrt(np.sum(np.square(self.hillEquation(self._data[:,0], self.thetaCmplx, self.ka) - self._data[:,1]))/obs)
    
    def fitFailed(self):
        self._fitSuccess = False
        self._ka = float('nan')
        self._kd = float('nan')
        self._thetaCmplx = float('nan')
        self._ngLL = float('nan')
        self._bic = float('nan')
        self._aic = float('nan')
        self._aicc = float('nan')
        self._rmse = float('nan')
        
    def predict(self, c):
        theta = self.hillEquation(c, self.thetaCmplx, self.ka)
        return theta
   
    def __str__(self):
        string = str("\nHill Model\nfitSuccess: %s\ninitial Parameter\ninitKa: %.3f\ninitKd: %.3f\ninitThetaCmplx: %.3f\nn: %i\nka: %.3f\nkd:%.3f\nthetaLgnd: %.3f\nthetaCmplx: %.3f\nlogL: %.3f\nbic: %.3f\naic: %.3f\naicc: %.3f\nrmse: %.3f\n" %(self.fitSuccess,
                                                                                                                                                                                                                                              self.initKa,
                                                                                                                                                                                                                                              self.initKd,
                                                                                                                                                                                                                                              self.initThetaCmplx,
                                                                                                                                                                                                                                              self.n,
                                                                                                                                                                                                                                              self.ka,
                                                                                                                                                                                                                                              self.kd,
                                                                                                                                                                                                                                              self.thetaLgnd,
                                                                                                                                                                                                                                              self.thetaCmplx,
                                                                                                                                                                                                                                              self.logL,
                                                                                                                                                                                                                                              self.bic,
                                                                                                                                                                                                                                              self.aic,
                                                                                                                                                                                                                                              self.aicc,
                                                                                                                                                                                                                                              self.rmse))
        return string
    
    def __del__(self):
        message  = str("removed instance of Hill from heap.")
        print(message)
       
# =============================================================================
# main function       
# =============================================================================
def main():
    dfRaw = pd.read_excel("/home/malkusch/PowerFolders/Met-HMM/modeling/Data/191108_HMM_FCSdata_MET_InlB321-ATTO647N.xlsx",
                          index_col = None,
                          header=0,
                          sheet_name = "Tabelle2")
    
    dfRaw = dfRaw[dfRaw["date"] == 120908]
    
    df = pd.DataFrame({"cLig": dfRaw.loc[1:,"MET [nM]"],
                      "tauD": dfRaw.loc[1:,"tauD1"]}).dropna()
    
    
    model = Hill()
    model.data = df.to_numpy()    
    model.thetaLgnd = 1.64
    model.n = 1
    model.initKd = 5.0
    model.initThetaCmplx = 4.0
    model.fitMLE()
    print(model)
    model.fitLSQ()
    print(model)
     
if __name__ == '__main__':
    main()