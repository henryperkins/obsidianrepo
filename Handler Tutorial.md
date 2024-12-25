### 2. Create EndpointHandler

After we have set up our environment, we can start creating your custom handler. The custom handler is a Python class (`EndpointHandler`) inside a `handler.py` file in our repository. The `EndpointHandler` needs to implement an `__init__` and a `__call__` method.

- The `__init__` method will be called when starting the Endpoint and will receive 1 argument, a string with the path to your model weights. This allows you to load your model correctly.
- The `__call__` method will be called on every request and receive a dictionary with your request body as a python dictionary. It will always contain the `inputs` key.

The first step is to create our `handler.py` in the local clone of our repository.

```
!cd distilbert-base-uncased-emotion && touch handler.py
```

In there, you define your `EndpointHandler` class with the `__init__` and `__call__ `method.

```python
from typing import Dict, List, Any

class EndpointHandler():
    def __init__(self, path=""):
        # Preload all the elements you are going to need at inference.
        # pseudo:
        # self.model= load_model(path)

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
       data args:
            inputs (:obj: `str` | `PIL.Image` | `np.array`)
            kwargs
      Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """

        # pseudo
        # self.model(input)
```

### 3. Customize EndpointHandler

Now, you can add all of the custom logic you want to use during initialization or inference to your Custom Endpoint. You can already find multiple [Custom Handler on the Hub](https://huggingface.co/models?other=endpoints-template) if you need some inspiration. In our example, we will add a custom condition based on additional payload information.

*The model we are using in the tutorial is fine-tuned to detect emotions. We will add an additional payload field for the date, and will use an external package to check if it is a holiday, to add a condition so that when the input date is a holiday, the model returns “happy” - since everyone is happy when there are holidays *🌴🎉😆 

First, we need to create a new `requirements.txt` and add our [holiday detection package](https://pypi.org/project/holidays/) and make sure we have it installed in our development environment as well.

```
!echo "holidays" >> requirements.txt
!pip install -r requirements.txt
```

Next, we have to adjust our `handler.py` and `EndpointHandler` to match our condition.

```python
from typing import Dict, List, Any
from transformers import pipeline
import holidays

class EndpointHandler():
    def __init__(self, path=""):
        self.pipeline = pipeline("text-classification",model=path)
        self.holidays = holidays.US()

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
       data args:
            inputs (:obj: `str`)
            date (:obj: `str`)
      Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """
        # get inputs
        inputs = data.pop("inputs",data)
        date = data.pop("date", None)

        # check if date exists and if it is a holiday
        if date is not None and date in self.holidays:
          return [{"label": "happy", "score": 1}]


        # run normal prediction
        prediction = self.pipeline(inputs)
        return prediction
```

### 4. Test EndpointHandler

To test our EndpointHandler, we can simplify import, initialize and test it. Therefore we only need to prepare a sample payload.

```python
from handler import EndpointHandler

# init handler
my_handler = EndpointHandler(path=".")

# prepare sample payload
non_holiday_payload = {"inputs": "I am quite excited how this will turn out", "date": "2022-08-08"}
holiday_payload = {"inputs": "Today is a though day", "date": "2022-07-04"}

# test the handler
non_holiday_pred=my_handler(non_holiday_payload)
holiday_payload=my_handler(holiday_payload)

# show results
print("non_holiday_pred", non_holiday_pred)
print("holiday_payload", holiday_payload)

# non_holiday_pred [{'label': 'joy', 'score': 0.9985942244529724}]
# holiday_payload [{'label': 'happy', 'score': 1}]
```