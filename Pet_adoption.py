import pandas as pd
import numpy as np

train_data = pd.read_csv('Feature_train.csv')
test_data = pd.read_csv('Feature_test.csv')



X_train = train_data.drop(['breed_category', 'pet_category'], axis=1)._get_numeric_data()
y_breed = pd.Series(train_data['breed_category'],dtype='int')
y_pet = train_data['pet_category']

X_test = test_data._get_numeric_data()

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(max_iter=69, verbose=3, max_depth=6, l2_regularization=2.05, learning_rate=0.075,
                                     early_stopping=False,warm_start=True)

hgb.fit(X_train, y_breed)
breed = hgb.predict(X_test)

X_train = pd.concat([X_train,train_data.breed_category],axis=1)
X_test = pd.concat([X_test,pd.DataFrame(breed,columns=['breed_category'])],axis=1)

hgb2 = HistGradientBoostingClassifier(max_iter=350,verbose=3,max_depth=6,l2_regularization= 2.05 ,learning_rate=0.075,early_stopping=False,warm_start=True,max_leaf_nodes=25)

hgb2.fit(X_train,y_pet)
pet = hgb2.predict(X_test)


res = pd.DataFrame(dict(pet_id=test_data.pet_id.values,
                        breed_category=breed,
                        pet_category=pet), dtype='int')

res.to_csv('Final_prediction.csv', index=False)
