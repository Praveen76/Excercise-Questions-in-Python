# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:58:44 2018

@author: praveenanwla
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
    
import datetime as dt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

cust = pd.read_csv("C:/Users/praveenanwla/Desktop/C_R2P/CSV files/customer.csv")

trans= pd.read_csv("C:/Users/praveenanwla/Desktop/C_R2Pn/CSV files/transactions.csv")

impression = pd.read_csv("C:/Users/praveenanwla/Desktop/C_R2P/CSV files/impressions.csv")


impression['channel']=impression['channel'].astype('category')
impression['id'] = pd.to_numeric(impression['id'])

##Data Transformation & Manipulation

adata1 =pd.DataFrame(pd.crosstab((impression.id,impression.date),impression.channel))

adata1=adata1.reset_index()
adata1.date= pd.to_datetime(adata1.date)

adata1.info()
print (adata1.dtypes)

#Convert DataFrame from wide format to long format
adata1 = pd.melt(adata1, id_vars=['id', 'date'], var_name='channel')

adata1['channel'] = adata1['channel'].astype('category')

np.unique(adata1['channel'])
adata1.columns = ['id', 'date','channel','impr']

NoImpressIds= list(set(cust.id) - set(impression.id))

dates = np.sort(np.unique(impression.date))
channels =np.unique(impression.channel)

Noimpressobs = pd.DataFrame({
                             'id': np.repeat(NoImpressIds , repeats=len(dates)* len(channels)),
                             'date' : np.repeat(np.repeat(dates , repeats = len(channels)), len(NoImpressIds)),
                            'channel' : np.repeat(channels , repeats = len(NoImpressIds)* len(dates)),
                            'impr' : np.repeat(0,len(dates)* len(NoImpressIds)* len(channels))
})


Noimpressobs = Noimpressobs.sort(['id'])
Noimpressobs.reset_index(drop = True, inplace = True)
adata1 = Noimpressobs.append(adata1)

adata1.describe()


#Pivot Data

adata = adata1.pivot_table(index=['id','date'],
                     columns='channel',
                    aggfunc=np.count_nonzero).reset_index().drop_duplicates(['id','date']).fillna(0)



adata.columns=adata.columns.droplevel() #Drop multilevels

adata.columns = ['id', 'date','direct', 'display', 'email', 'email.holdout', 'social']

#Cross Verify lengths
sum(adata.direct) == len(impression.channel[impression.channel=="direct"]) 

###Transactions File   
trans= pd.read_csv("C:/Users/praveenanwla/Desktop/C_R2P/CSV files/transactions.csv")

 
atrans = trans.groupby(['id','date']).count().reset_index()

##Quick check 
sum(atrans_1.direct) == len(impression.channel[impression.channel=="direct"])

atrans=atrans.reset_index()
atrans.id = atrans['id'].astype(int)
atrans.date= pd.to_datetime(atrans.date)

#Merge both files
adata = pd.merge(adata ,atrans , on=['id','date'] , how='outer' )
adata1['trans']=adata1['trans'].fillna(0)

adata.date= pd.to_datetime(adata.date)

#Remove first and last days ( which are incomplete )

adata= adata[(adata.date !=" 2016-12-31 ") & (adata.date != " 2017-02-28 ") & (adata.date != " 2017-02-27")]

# Add customer info from cust table

adata = pd.merge(adata ,cust , on= 'id')

# #Tidy up column names 
# Note:- Give column names according to dataset in below line of code

adata.columns = [" direct ", " display ", " email ", " email . holdout ", " social ",
                  " trans ", " past . purchase ", " has . email ", " has. direct "]

del adatal , atrans 
adata.describe()

adata[(adata.date !=" 2017-01-03")].describe()

#==============================================================================
# #Model building
#Add direct + display + email + social as IVs against Dv trans
#==============================================================================


X = adata.iloc[:,[2,3]].values   ### Assign columns  location respectively
Y = adata.iloc[:,4].values     ### Assign DV column  location accordingly


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, Y , test_size = 1/3, random_state = 0)

# Fitting Simple Logistic Regression to the Training set

classifier = LogisticRegression(random_state = 0 , verbose=1 ,solver = 'sag' )
Model = classifier.fit(X_train, y_train)

Model.summary()
 
# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



#==============================================================================
# Because we have a user-level model, we can bring in user characteristics.
#Add direct + display + email + social + past.purchase as IVs against Dv trans
#==============================================================================


