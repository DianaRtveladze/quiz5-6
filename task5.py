import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# მონაცემთა ჩატვირთვა
df = pd.read_csv('pizza_sales.csv')

# დაყოფა მონაცემების სასწავლო და ტესტირების სეტებად
X_train, X_test, y_train, y_test = train_test_split(df[['x1', 'x2']], df['y'], test_size=0.2, random_state=0)

# მოდელის შექმნა და მონაცემებზე მორგება 
model = DecisionTreeClassifier().fit(X_train, y_train)

# აუთფუთის ვარაუდი
y_pred = model.predict(X_test)

# გამოთვლა მოდელის ეფექტურობის
r_sq = model.score(X_test, y_test)
print(f'The accuracy of the model is {r_sq:.2f}.')

