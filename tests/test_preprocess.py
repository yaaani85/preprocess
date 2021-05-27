
from ..brain_preprocessing.py import preprocess

#local testing
def test_preprocess():
  assert preprocess(True, True) == "Preprocessed data"
  
