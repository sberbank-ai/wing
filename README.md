# О проекте
Данный проект реализует sklearn-based Transformer для Weight of evidence преобразования.  
Это одно из наиболее удобных и результативных преобразований для логистической регрессии.

# Как использовать?

1. Установите модуль:
```bash
python setup.py install
```
2. Импортируйте объект и работайте с ним как sklearn Transformer:
```python

import pandas as pd
from wingilya import WingsOfEvidence
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X,y = make_classification(n_samples=10000,n_features=10,n_informative=2,random_state=42)

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.3,random_state=42)

column_names = ['X%i'%i for i in range(10)]

D_train = pd.DataFrame(X_train,columns = column_names)
D_test = pd.DataFrame(X_test,columns = column_names)

# Для высокой скорости используйте optimizer='ilya-binning', для высокого качества используйте optimizer='full-search'
# Параметры, которые оказывают влияние на биннинг при optimizer='ilya-binning' обозначены ниже
# Все остальные параметры используются при optimizer='full-search'
wing = WingsOfEvidence(bin_minimal_size=0.05, bin_size_increase=0.05, is_monotone=False)
log = LogisticRegression()

pipe = Pipeline(steps=
    [
        ('wing',wing),
        ('log',log)
    ]
)

pipe.fit(D_train,y_train)

test_proba = pipe.predict_proba(D_test)


```
