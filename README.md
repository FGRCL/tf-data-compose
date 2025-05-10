# TF Data Compose
TF Data Compose allows you to create scikit-learn-like data preprocessing pipelines using tensorflow.
Data preprocessing pipelines can be composed to highly reusable, configurable, and composable preprocessing functions.

```python
dataset = Dataset.from_tensor_slices([-10,-9,NaN,-8,-7,-6,-5,NaN,-4,-3,-2,-1,0,1,NaN,2,3,4,5,6,7,8,9,10])
print(list(dataset.as_numpy_iterator())) # [-10.0, -9.0, nan, -8.0, -7.0, -6.0, -5.0, nan, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

class CleanData(Pipeline):
    def __init__(self):
        super().__init__([
            RemoveNone(),
            RemoveOutOfRange(-5, 5),
        ])

positiveOdd = Pipeline([
    CleanData(),
    FilterSign(positive=True),
    FilterOddness(even=False)
])(dataset)
print(list(positiveOdd.as_numpy_iterator())) # [1.0, 3.0]

negativeEven = Pipeline([
    CleanData(),
    FilterSign(positive=False),
    FilterOddness(even=True)
])(dataset)
print(list(negativeEven.as_numpy_iterator())) # [-4.0, -2.0, 0.0]
```

View the [docs](https://fgrcl.github.io/tf-data-compose/index.html) for more information.
