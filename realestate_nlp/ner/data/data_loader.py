import json, gzip, gc
import pandas as pd
from datetime import datetime
from pathlib import Path

from ...common import class_extensions
from ...common.run_config import home, bOnColab



def load_full_listing_df(data_dir, dedup_remark=False, res_condo_only=True):

  uat_listing_df = pd.read_pickle(data_dir/'full_listing_presentation_b_df.pickle.gz', compression='gzip')
  print(f'Loaded full_listing_presentation_b_df from AVM UAT dev: {len(uat_listing_df)}')

  floorSpace_cols = [c for c in uat_listing_df.columns if 'floor' in c]
  interior_cols = [c for c in uat_listing_df.columns if 'sizeInterior' in c]

  es_pickle_path = data_dir/'listing_es_df_pickles'
  listing_es_df = pd.read_feather(es_pickle_path/'listing_es_df')
  print(f'Loaded listing_es_df from image tagging: {listing_es_df.shape[0]}')

  synthetic_sentences = [
    {"jumpId": "syn00000001", "listingType": "RES", "remarks": "This house has a 1000 SqFt+ Garage."},
    {"jumpId": "syn00000002", "listingType": "RES", "remarks": "This house has 1000 SqFt + Garage."},
    {"jumpId": "syn00000003", "listingType": "RES", "remarks": "This house has 1000 SqFt plus Garage."},
    {"jumpId": "syn00000004", "listingType": "RES", "remarks": "This big deck on both main level has 510 sq ft!"},
    {"jumpId": "syn00000005", "listingType": "RES", "remarks": "This balcony on both main level has 510 sq ft!"},
    {"jumpId": "syn00000006", "listingType": "RES", "remarks": "This balcony has 510 sq ft!"},
    {"jumpId": "syn00000007", "listingType": "RES", "remarks": "A private 400 cubic ft of storage."},
    {"jumpId": "syn00000008", "listingType": "RES", "remarks": "This house has 900 sf terr space."},
    {"jumpId": "syn00000009", "listingType": "RES", "remarks": "7000 sqft cowork space."},
    {"jumpId": "syn00000010", "listingType": "RES", "remarks": "The deck on both main level has 600 sq ft!"},
    {"jumpId": "syn00000011", "listingType": "RES", "remarks": "This deck has 750 sq ft!"},
    {"jumpId": "syn00000012", "listingType": "RES", "remarks": "the ground level patio offers 425 sq ft"},
    {"jumpId": "syn00000013", "listingType": "RES", "remarks": "The patio on both main level has 700 sq ft!"},

    {"jumpId": "syn00000014", "listingType": "RES", "remarks": "The house has Approx5500 sq ft living space!"},
    {"jumpId": "syn00000015", "listingType": "RES", "remarks": "Extraordinary Unit 2+1, 990 Sqft With Large Balcony."},
    {"jumpId": "syn00000016", "listingType": "RES", "remarks": "This bungalow with a garage is 2650 sq ft."},
    {"jumpId": "syn00000017", "listingType": "RES", "remarks": "This house with a large deck is 2650 sq ft."},
    {"jumpId": "syn00000018", "listingType": "RES", "remarks": "This semi-detached with a large patio is 1750 sq ft."},
    {"jumpId": "syn00000019", "listingType": "RES", "remarks": "The main house has 5 bedrooms, 4 baths, including a large patio, boasts approx. 4200 sqft."}

  ]

  synthetic_df = pd.DataFrame(columns=listing_es_df.columns).append(synthetic_sentences, ignore_index=True)

  full_listing_df = pd.concat([
                              listing_es_df, 
                              uat_listing_df[['jumpId', 'listingType', 'remarks', 'beds', 'baths'] + floorSpace_cols + interior_cols + ['address', 'provState', 'price']],
                              synthetic_df
                              ], axis=0, ignore_index=True)

  # drop those without remarks
  idx = full_listing_df.q_py("remarks.isnull()").index
  full_listing_df.drop(index=idx, inplace=True)

  if res_condo_only:
    idx = full_listing_df.q_py("~listingType.isin(['RES', 'CONDO'])").index
    full_listing_df.drop(index=idx, inplace=True)

  full_listing_df.drop_duplicates(subset='jumpId', keep='last', inplace=True)
  if dedup_remark:
    full_listing_df.drop_duplicates(subset='remarks', keep='last', inplace=True)
    
  full_listing_df.defrag_index(inplace=True)
  print(f'len(full_listing_df): {full_listing_df.shape[0]}')

  return full_listing_df

def load_remarks_ner_all_df(data_dir):
  remarks_ner_all_df = []
  for f in data_dir.lf('remarks_*_df.pickle'):
    remarks_ner_all_df.append(pd.read_pickle(f))

  remarks_ner_all_df = pd.concat(remarks_ner_all_df, axis=0, ignore_index=True)
  return remarks_ner_all_df

def load_avm_prod_snapshot(snapshot_date=None, download_from_gs=True) -> pd.DataFrame:
  '''
  Load the raw json snapshot during AVM monitoring and return it as a dataframe.
  '''

  if snapshot_date is None:
    snapshot_date = datetime.today().date().strftime("%Y_%m_%d")
  else:
    snapshot_date = snapshot_date.strftime('%Y_%m_%d')

  print(f'{"Running on colab" if bOnColab else "Running on local machine"}, saving to /content')

  # copy snapshot from gcs to local
  if bOnColab:
    data_dir = Path('/content')
  else:
    data_dir = Path('/Users/kelvinchan/tmp')

  if download_from_gs:
    from google.cloud import storage
    storage_project_id = 'royallepage.ca:api-project-267497502775'
    storage_client = storage.Client(project=storage_project_id)
    storage_bucket = storage_client.get_bucket('ai-tests')

    snapshot_json_files = [f.name for f in storage_bucket.list_blobs(prefix=f'AVMDataAnalysis/quickquote_prod_snapshot_{snapshot_date}')]
    for f in snapshot_json_files:
      print(f'Downloading {Path(f).name} from gcs')
      blob = storage_bucket.blob(f)
      blob.download_to_filename(f'{data_dir/Path(f).name}')    # f'/content/{Path(f).name}'

  listing_dfs = []
  for f in Path('/content').lf(f'quickquote_prod_snapshot_{snapshot_date}*.json.gz'):
    print(f'Processing {f}')
    with gzip.open(f, 'r') as fp:
      payload = json.load(fp)
      
    listing_df = pd.DataFrame(payload)
    listing_df.listingDate = pd.to_datetime(listing_df.listingDate)
    listing_df.lastUpdate = pd.to_datetime(listing_df.lastUpdate, format="%y-%m-%d:%H:%M:%S")
    listing_df.addedOn = pd.to_datetime(listing_df.addedOn, format="%Y-%m-%dT%H:%M:%S")

    idx = listing_df.q_py("lastUpdate < '2021-12-01'").index
    listing_df.drop(index=idx, inplace=True)
    listing_df.listingType = listing_df.listingType.apply(lambda a: a[-1])  # e.g. [listingType, RES] => 'RES'

    listing_dfs.append(listing_df)
  
  listing_df = pd.concat(listing_dfs, axis=0, ignore_index=True)
  del listing_dfs
  gc.collect()

  listing_df.drop_duplicates(subset='jumpId', inplace=True, ignore_index=True)

  return listing_df

