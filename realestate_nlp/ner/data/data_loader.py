import pandas as pd

def load_full_listing_df(data_dir, dedup_remark=False):

  uat_listing_df = pd.read_pickle(data_dir/'full_listing_presentation_b_df.pickle.gz', compression='gzip')
  print(f'Loaded full_listing_presentation_b_df: {len(uat_listing_df)}')

  floorSpace_cols = [c for c in uat_listing_df.columns if 'floor' in c]
  interior_cols = [c for c in uat_listing_df.columns if 'sizeInterior' in c]

  es_pickle_path = data_dir/'listing_es_df_pickles'
  listing_es_df = pd.read_feather(es_pickle_path/'listing_es_df')
  print(f'len(listing_es_df): {listing_es_df.shape[0]}')  # 319540, 380351, 428256, 447473

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

  full_listing_df.drop_duplicates(subset='jumpId', keep='last', inplace=True)
  if dedup_remark:
    full_listing_df.drop_duplicates(subset='remarks', keep='last', inplace=True)
  full_listing_df.reset_index(drop=True, inplace=True)
  print(f'len(full_listing_df): {full_listing_df.shape[0]}')

  return full_listing_df

def test_q_py(df, q_str):
  return df.q_py(q_str)