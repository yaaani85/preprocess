
from . import brane_preprocessing

#local testing
def test_preprocess():
  assert brane_preprocessing(True, True) == "Preprocessed data"
  
