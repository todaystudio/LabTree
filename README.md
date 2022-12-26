# Классификация с использованием метода дерева решений
Классификация в машинном обучении — это процесс группирования объектов по категориям на основе предварительно классифицированного тренировочного набора данных.

Классификация относится к алгоритмам контролируемого обучения. В них используется категоризация тренировочных данных, чтобы рассчитать, с какой вероятностью новый объект попадёт в одну из заданных категорий. Известный всем пример алгоритмов классификации — это сортировка входящих электронных писем как спам или не спам. 

Дерево решений (decision tree) – один из наиболее часто и широко используемых алгоритмов контролируемого машинного обучения, который может выполнять задачи, как регрессии, так и классификации. Интуиция, лежащая в основе алгоритма decision tree, проста, но при этом очень эффективна. Для каждого атрибута в наборе данных алгоритм дерева решений формирует узел, в котором наиболее важный атрибут помещается в корневой узел. Для оценки мы начинаем с корневого узла и продвигаемся вниз по дереву, следуя за соответствующим узлом, который соответствует нашему условию или «решению». Этот процесс продолжается до тех пор, пока не будет достигнут конечный узел, содержащий прогноз или результат дерева решений.

### Преимущества decision tree:

* Использование дерева решений для прогнозного анализа дает несколько преимуществ: decision tree могут использоваться для прогнозирования как непрерывных, так и дискретных значений, т.е. они хорошо работают, как для задач регрессии, так и для классификации. 

* Для их обучения требуется относительно меньше усилий.

* Их можно использовать для классификации нелинейно разделимых данных. 

* Они очень быстрые и эффективные по сравнению с KNN и другими алгоритмами классификации.

### Пример метода decision tree

Ниже приведён пример реализации модели классификации с использованием метода дерева решений на языке Python и библиотеки Scikit-Learn. Для начала нам потребуются библиотеки pandas и scikit-learn.

```python
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
```
Далее мы осуществляем загрузку датасета.

```python
dataset = pd.read_csv('D:\\bill_authentication.csv')
```
```python
X = dataset.drop('Class', axis=1)
y = dataset['Class']
```

Здесь переменная X содержит все столбцы из набора данных, кроме столбца «Класс», который является меткой. Переменная Y содержит значения из столбца «Класс». Переменная X – это наш набор атрибутов, а переменная Y содержит соответствующие метки.

Затем, мы случайно разделяем наши данные на обучающие и тестовые наборы.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
```

Параметр test_size указывает соотношение набора тестов, которое мы используем для разделения 20% данных на набор тестов и 80% для обучения

После этого мы выполняем обучение алгоритма:

```python
y_pred = classifier.predict(X_test)
```

Далее производим оценку точности алгоритма. Для задач классификации часто используются такие показатели, как матрица неточностей, точность, отзыв и оценка F1.

```python
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Это даст следующую оценку:

![Alt-текст](https://avatars1.githubusercontent.com/u/5384215?v=3&s=460 "Орк")

Из матрицы неточностей видно, что из 275 тестовых примеров наш алгоритм неправильно классифицировал только 5. Это точность 98,5%.

Далее с помощью библиотеки graphviz визуализируем само дерево решений

```python
import graphviz 
dot_data = tree.export_graphviz(classifier, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("tree")
```

![Alt-текст](https://avatars1.githubusercontent.com/u/5384215?v=3&s=460 "Орк")

Модули для построения и исследования деревьев решений входят в состав множества аналитических платформ. Это удобный инструмент, применяемый в системах поддержки принятия решений и интеллектуального анализа данных. Круг их использования постоянно расширяется, а деревья решений постепенно становятся важным инструментом управления бизнес-процессами и поддержки принятия решений.