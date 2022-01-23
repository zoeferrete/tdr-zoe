import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from matplotlib import pyplot as plt
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
random_state = 42

# Creación de la base de datos
names = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'class']
df = pd.read_csv(url, names = names)

# Conversión de clases a íntegros (Iris-setosa=0; Iris-versicolor=1; Iris-virginica=2)
df["class"] = LabelEncoder().fit_transform(df["class"])

# Definición X_train, X_test, Y_train y Y_test.
X = df.drop("class", axis = 1)
Y = df["class"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state = random_state)

# Creación modelo neuronal.
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, Y_train.values.ravel())
predictions = mlp.predict(X_test)
print(classification_report(Y_test,predictions))

# Diagrama de decisiones.
model_tree = DecisionTreeClassifier()
model_tree.fit(X_train, Y_train)
fig = plt.figure(figsize = (25, 20))
_ = tree.plot_tree(model_tree,
                feature_names=["sepalLength", "sepalWidth", "petalLength", "petalWidth"], 
                class_names=["I. setosa", "I. versicolor", "I. virginica"], 
                filled=True, 
                rounded = True)
