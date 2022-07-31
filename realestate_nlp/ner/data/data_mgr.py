from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union
import json

def get_jumpIds_from_remarks_json(filename) -> List:
  '''
  Loads a json file (e.g. Remarks_NER_46.json) with remarks and returns a list of jumpIds.
  '''

  listings = []
  with open(filename) as f:
    for line in f:
      listings.append(json.loads(line))

  listings = [listing['jumpId'] for listing in listings]
  return listings