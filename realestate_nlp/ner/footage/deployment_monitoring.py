from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union
from pathlib import Path

import pandas as pd
import re

from realestate_nlp.common.run_config import home, bOnColab

def analyse_logs(logfile: str) -> pd.DataFrame:
  logfile = Path(logfile)

  monitored_listings = []
  prediction_count = 0
  with open(logfile, 'r') as f:
    while True:
      line = f.readline()
      if 'WARNING - MONITOR' in line:
        warning_content = re.compile(r'.*?WARNING - MONITOR:\s+(.*?)$').match(line).group(1)
        try:
          warning_content = eval(warning_content)
          monitored_listings.append(warning_content)
        except:
          print(f'{warning_content}: not a dict')
      
      if 'Prediction served for' in line: prediction_count += 1
        
      if not line: break

  monitor_listing_df = pd.DataFrame(monitored_listings)
  return monitor_listing_df
