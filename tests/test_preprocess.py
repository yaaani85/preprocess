
from ..brain_preprocessing.py import preprocess

#local testing
def test_preprocess():
  assert preprocess(true, true) == "Preprocessed data"
  
