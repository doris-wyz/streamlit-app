import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
import category_encoders as ce
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.backend import exp
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

import utils
import shap

class Anomaly_Detect:

    def __init__(self,data,outliers_fraction = .05):
        self.data = data.copy()
        self.data_org = data.copy()
        self.outliers_fraction = .05
        #droping null values 
        #self.data = self.data.dropna()

        self.target_scaler = StandardScaler()
        self.cont_scaler = StandardScaler()
        self.dropped_cols = []
        


    def prepare_data(self,target,timestamp,continuous,categories,task=None):
        
        self.task = task
        
        if target:
            self.target_scaler = StandardScaler()
            self.data[[target]] = self.target_scaler.fit_transform(self.data[[target]])
            self.target = target

        if continuous:
            for col in continuous:
                self.data[col].fillna(self.data[col].mean())
            self.cont_scaler = StandardScaler()
            self.data[continuous] = self.cont_scaler.fit_transform(self.data[continuous])

        if categories:
            for cat in categories:
                self.data[cat].fillna(self.data[cat].mode()[0])
            self.cat_encoders = ce.OneHotEncoder(cols=categories,use_cat_names=True)
            self.data = self.cat_encoders.fit_transform(self.data)

        if timestamp:
            for ts in timestamp:
                    self.fe_timestamps(ts,task)
        
        for col in self.data.columns:
                if (self.data[col].nunique() == 1) or ((task == 'cluster') and(self.data[col].nunique() == self.data.shape[0])):
                    self.data.drop(col,axis=1,inplace=True)
                    self.dropped_cols.append(col)
                    print("columnn dropped ",col)

        return self.data.columns, self.dropped_cols
        
        

    def fe_timestamps(self,column_name,task):

        #convert to timestamp 
        print("Column Name",column_name)
        self.data[column_name] = pd.to_datetime(self.data[column_name])

        # If Timeseries
        if task == "timeseries":
            self.data['time_epoch'] = list(range(self.data.shape[0]))
            self.data.sort_values(by=[column_name], inplace=True)
        
        else:
        # Extract features
            self.data['year'] = self.data[column_name].dt.year
            self.data['month'] = self.data[column_name].dt.month
            self.data['day'] = self.data[column_name].dt.day
            
            self.data['hour'] = self.data[column_name].dt.hour
            self.data['minute'] = self.data[column_name].dt.minute

            # Clean values
            new_cols = ['year','month','day','hour','minute']
            for col in new_cols:
                if self.data[col].nunique() == 1:
                    self.data.drop(col,axis=1)

    def process_isolation_forest(self,contamination=.05):

        iso_col = 'iso_prediction'
        #if iso_col not in self.data.columns:
        iso_for = IsolationForest(random_state=0,contamination=contamination)
        if self.task == 'cluster':
            
            self.data_org['iso_prediction']=iso_for.fit_predict(self.data)
            explainer = shap.TreeExplainer(iso_for)
            shap_values = explainer.shap_values(self.data)
            anomaly_idx = self.data_org[self.data_org.iso_prediction == -1].index[0]
            if not anomaly_idx:
                anomaly_idx = 0
            print("Anomaly idx",anomaly_idx)
            p = shap.force_plot(explainer.expected_value,shap_values[anomaly_idx,:], self.data.iloc[anomaly_idx,:])
            sum_plot = shap.summary_plot(shap_values,self.data)
            sum_plot_box = shap.summary_plot(shap_values,self.data,plot_type='bar')
            return  p, sum_plot, sum_plot_box
            
        elif self.task == 'timeseries':
            
            iso_for.fit(self.data[[self.target]])
            self.data_org['iso_prediction'] = iso_for.predict(self.data[[self.target]])

            fig, ax = plt.subplots()
            a = self.data.loc[self.data_org['iso_prediction'] == -1, ['time_epoch', self.target]] #anomaly

            ax.plot(self.data['time_epoch'], self.data[self.target], color='blue')
            ax.scatter(a['time_epoch'],a[self.target], color='red')

            return fig

    def process_elliptical_envelope(self,contamination=.05):
        iso_col = 'iso_prediction'
        #if iso_col not in self.data.columns:
        ell_env = EllipticEnvelope(random_state=0,contamination=contamination)
        if self.task == 'cluster':
            self.data_org['eenv_prediction']=ell_env.fit_predict(self.data)
            
            explainer = shap.KernelExplainer(ell_env.decision_function,data=self.data[:10])
            shap_values = explainer.shap_values(self.data[:10])
            p=shap.force_plot(explainer.expected_value,shap_values[0,:], self.data.iloc[0,:])
            sum_plot = shap.summary_plot(shap_values,self.data[:10])
            return p,sum_plot
            
        elif self.task == 'timeseries':
            ell_env.fit(self.data[[self.target]])
            self.data_org['eenv_prediction'] = ell_env.predict(self.data[[self.target]])

            fig, ax = plt.subplots()
            a = self.data.loc[self.data_org['eenv_prediction'] == -1, ['time_epoch', self.target]] #anomaly

            ax.plot(self.data['time_epoch'], self.data[self.target], color='blue')
            ax.scatter(a['time_epoch'],a[self.target], color='red')

            return fig

    def process_svm(self,contamination=.05):
        svm_col ="svm_prediction"
        if self.task == "timeseries":
            svm_mdl = OneClassSVM(nu=0.95 * contamination,kernel="linear")
            svm_mdl.fit(self.data[[self.target]])
            self.data_org['svm_prediction']= svm_mdl.predict(self.data[[self.target]])

            fig, ax = plt.subplots()
            a = self.data.loc[self.data_org[svm_col] == -1, ['time_epoch', self.target]] #anomaly

            ax.plot(self.data['time_epoch'], self.data[self.target], color='blue')
            ax.scatter(a['time_epoch'],a[self.target], color='red')

            return fig

        elif self.task == "cluster":
            svm_mdl = OneClassSVM(nu=0.95 * contamination,kernel="linear")
            svm_mdl.fit(self.data)
            self.data_org['svm_prediction']= svm_mdl.predict(self.data)

            explainer = shap.KernelExplainer(svm_mdl.decision_function,data=self.data[:10])
            shap_values = explainer.shap_values(self.data[:10])
            p=shap.force_plot(explainer.expected_value,shap_values[0,:], self.data.iloc[0,:])
            sum_plot = shap.summary_plot(shap_values,self.data[:10])
            return p,sum_plot
    
    def build_lstm(self,num_cols):
        model = Sequential()
        model.add(LSTM(units = 50, return_sequences = True, input_shape = (num_cols, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 50))
        model.add(Dropout(0.2))
        model.add(Dense(units = 1))

        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        return model
        

    def process_rnn(self,win_size=8):
        X_train = []
        y_train = []
        target = self.target

        d = self.data[target].values
        for i in range(win_size, len(d)):
            X_train.append(d[i-win_size:i])
            y_train.append(d[i]) 

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        self.lstm = self.build_lstm(X_train.shape[1])

        #Early stopping
        early_stopping = EarlyStopping(patience=3)
        self.lstm.fit(X_train,y_train,epochs=5,callbacks=[early_stopping])

        self.data_org['lstm_predict'] = np.nan

        self.data_org.loc[win_size:,'lstm_predict'] = self.target_scaler.inverse_transform(self.lstm.predict(X_train))

        d = self.data_org.dropna()

        self.data_org['lstm_predict'].fillna(0,inplace=True)
        self.data_org['is_anomaly'] = abs(self.data_org[self.target] - self.data_org['lstm_predict']) / self.data_org[self.target]
        self.data_org['is_anomaly'] = self.data_org['is_anomaly'].apply(lambda x : 1 if x<.1 else -1)

        fig, ax = plt.subplots()
        ax.plot(range(d.shape[0]), d[self.target],color='blue')
        ax.plot(range(d.shape[0]), d['lstm_predict'],color='yellow')
        plt.plot()

        
        return fig