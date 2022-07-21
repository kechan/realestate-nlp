import pandas as pd
from functools import partialmethod
pd.DataFrame.q_py = partialmethod(pd.DataFrame.query, engine='python')
pd.DataFrame.defrag_index = partialmethod(pd.DataFrame.reset_index, drop=True)

from pathlib import Path
Path.ls = lambda x: list(x.iterdir())
Path.lf = lambda pth, pat='*': list(pth.glob(pat))
Path.rlf = lambda pth, pat='*': list(pth.rglob(pat))