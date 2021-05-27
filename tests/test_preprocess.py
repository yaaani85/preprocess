
from . import brain_preprocessing

#local testing
def test_preprocess():
  assert preprocess(True, True) == "Preprocessed data"
  
