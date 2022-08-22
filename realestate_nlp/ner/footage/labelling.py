# all things to do with weak or semi-weak labelling
import re
import pandas as pd
from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union

import tensorflow as tf
import numpy as np
# from transformers import AutoTokenizer, DistilBertTokenizer, TFDistilBertModel, TFDistilBertForTokenClassification

NUM_REGEX = "[0-9]*?[-+.,]?[0-9]*[.]?[0-9]+\+*"   # 1[,]23..[+]

ANY_SPACE_PLUS_REGEX = '[\s\+]*'

TOTAL_REGEX = '(?i:total)'

FT_REGEX = "(?i:ft\.*)"   # ft[.]
SQFT_REGEX = "(?i:sq\.*\s*/*ft\.*)"   # sq[.][/]ft[.]
FTSQ_REGEX = "(?i:ft\.*\s*/*sq\.*)"   # ft[.][/]sq[.]
SQUAREFEET_REGEX = "(?i:square\.*\s*feet)"   # square[.][ ]feet
SQFEET_REGEX = "(?i:sq\.*\s*feet)"   # sq[.][ ]feet
SQUAREFOOT_REGEX = "(?i:square\.*\s*foot)"   # square[.][ ]foot
SQFOOT_REGEX = "(?i:sq\.*\s*foot)"           # sq[.][ ]foot
SF_REGEX = "(?i:s\.*\s*/*f\.*)"     # s[.][ ]f[.]
SQF_REGEX = "(?i:sq\.*f*t*)"           # sq[.]f or sq[.]t

# M_REGEX = "(?i:m\.*)"       # this causes too many false +ve
SQM_REGEX = "(?i:sq\.*\s*/*m\.*)"
MSQ_REGEX = "(?i:m\.*\s*/*sq\.*)"
SQUAREMETER_REGEX = "(?i:square\.*\s*meter)"
SQMETER_REGEX = "(?i:sq\.*\s*meter)"
SQUAREMETRE_REGEX = "(?i:square\.*\s*metre)"
SQMETRE_REGEX = "(?i:sq\.*\s*metre)"

UNIT_IMP_REGEXES = [FT_REGEX, SQFT_REGEX, FTSQ_REGEX, SQUAREFEET_REGEX, SQFEET_REGEX, SQUAREFOOT_REGEX, SQFOOT_REGEX, SF_REGEX, SQF_REGEX]
UNIT_METRIC_REGEXES = [#M_REGEX, 
                       SQM_REGEX, MSQ_REGEX, SQUAREMETER_REGEX, SQMETER_REGEX, SQUAREMETRE_REGEX, SQMETRE_REGEX]

# generate combinatorial regexes
all_regexes = []
for unit_imp_regex in UNIT_IMP_REGEXES:
  all_regexes.append(f"({NUM_REGEX}{ANY_SPACE_PLUS_REGEX}{unit_imp_regex})")

for unit_imp_regex in UNIT_IMP_REGEXES:
  all_regexes.append(f"({NUM_REGEX}{ANY_SPACE_PLUS_REGEX}{TOTAL_REGEX}{ANY_SPACE_PLUS_REGEX}{unit_imp_regex})")

for unit_metric_regex in UNIT_METRIC_REGEXES:
  all_regexes.append(f"({NUM_REGEX}{ANY_SPACE_PLUS_REGEX}{unit_metric_regex})")

for unit_metric_regex in UNIT_METRIC_REGEXES:
  all_regexes.append(f"({NUM_REGEX}{ANY_SPACE_PLUS_REGEX}{TOTAL_REGEX}{ANY_SPACE_PLUS_REGEX}{unit_metric_regex})")

all_regexes_2 = []    # capturing separately the number and unit.

for unit_imp_regex in UNIT_IMP_REGEXES:
  all_regexes_2.append(f"({NUM_REGEX}){ANY_SPACE_PLUS_REGEX}({unit_imp_regex})")

for unit_imp_regex in UNIT_IMP_REGEXES:
  all_regexes_2.append(f"({NUM_REGEX}){ANY_SPACE_PLUS_REGEX}{TOTAL_REGEX}{ANY_SPACE_PLUS_REGEX}({unit_imp_regex})")

for unit_metric_regex in UNIT_METRIC_REGEXES:
  all_regexes_2.append(f"({NUM_REGEX}){ANY_SPACE_PLUS_REGEX}({unit_metric_regex})")

for unit_metric_regex in UNIT_METRIC_REGEXES:
  all_regexes_2.append(f"({NUM_REGEX}){ANY_SPACE_PLUS_REGEX}{TOTAL_REGEX}{ANY_SPACE_PLUS_REGEX}({unit_metric_regex})")

# test strings for regex
test_strings = [
                'Nice 500 sqt condo.',
                'Sunny 500 s/f living space with 200 Sqf basement',
                'this nice condo has 1700 Total sq ft',
                '3 beds, and a 1600 sq meter deck space',
                'Roomy 45sq m(484 sq ft) bachelor suite with huge balcony',
                'this house is 1,200sqft huge.',
                '2 storey home boats 2183sq.ft. of living space',

                '3 sq ft cat house', '4.5 sq ft dog house',                
                '10 sq ft cage home', '200 sq ft tiny house',                                
                '300.5 sq ft tiny house',

                '10,000 square foot mansion',
                '10,000+square foot mansion',

                '619+99 Sq Ft Balcony',
                '2000 sq foot semi',
                '2,500 sq feet semi-detached',
                'this house has 1700+sqft living space.',                
                '3 beds, 2 baths, and a 2000 ft deck space',
                '3 beds, 2 baths, and a 5400 Sq Ft deck space',
                '3 beds, 2 baths, and a 5400 Sq/Ft deck space',
                '3 beds, 2 baths, and a 5400 sq. ft. deck space',
                '3 beds, 2 baths, and a 5400 Sq. Ft. deck space',
                '3 beds, 2 baths, and a 5400+Sq. Ft. deck space',
                '3 beds, 2 baths, and a 5400 Sq./Ft. deck space',
                '3 beds, 2 baths, and a 5400.01 sqft deck space',
                '3 beds, 2 baths, and a 5400 Sqft deck space',
                '3 beds, 2 baths, and a 5400 SqFt deck space',
                '3 beds, 2 baths, and a 5,400.01 SqFt. deck space',
                '3 beds, 2 baths, and a 5,400 square feet deck space',
                '3 beds, 2 baths, and a 5,400+square feet deck space',
                '3 beds, 2 baths, and a 5,400 sf deck space',
                '3 beds, 2 baths, and a 5,400 Sf deck space',
                '3 beds, 2 baths, and a 5,400 S.F. deck space',                
                '3 beds, 2 baths, and a 5,400+S.F. deck space',                
                '3 beds, 2 baths, and a 1600+ sq ft deck space',                
                '3 beds, 2 baths, and a 5,400 S F deck space',
                '3 beds, 2 baths, and a 5,400+ S F deck space',
                'this house floor space is 1,200 ft sq. and terrace space of 100 ft sq.',   #[('ft sq', 1, 'size_unit_imp'), ('ft sq', 0, 'size_unit_imp'), ('1,200', 0, 'size_internal'), ('100', 0, 'size_external')]
                'a 2000 ft. deck space, with 2,000 Sq ft patio space'
]


def get_html_w_number_or_sq_ft_colored(data: Union[pd.DataFrame, pd.Series, str], index=False, color="red", excl_lone_num=False) -> str:
  if isinstance(data, pd.DataFrame):
    html_str = data.to_html(index=index)
    head_html_str = html_str[:39]      # don't run regex sub over the html "head"
    body_html_str = html_str[39:]
  elif isinstance(data, pd.Series):
    html_str = data.to_frame().to_html()
    head_html_str = html_str[:39]
    body_html_str = html_str[39:]
  elif isinstance(data, str):
    head_html_str = ''
    body_html_str = data
  else:
    pass

  # color a number for by imperial unit of sq ft
  # body_html_str = re.sub(r"([0-9]?[-+.,]?[0-9]+[.]?[0-9]+\+*\s+(?i:ft\.*))", fr'<font color="{color}"><b>\1</b></font>', body_html_str)
  for regex_str in all_regexes:
    body_html_str = re.sub(fr"{regex_str}", fr'<font color="{color}"><b>\1</b></font>', body_html_str)

  if not excl_lone_num:
    body_html_str = re.sub(r"(\d+)", fr'<font color="{color}"><b>\1</b></font>', body_html_str)
    # body_html_str = re.sub(r"(?<!>)(\d+)(?!<)", fr'<font color="{color}"><b>\1</b></font>', body_html_str)

  return head_html_str + body_html_str

def has_number_followed_by_unit(remark: str) -> bool:

  for regex_str in all_regexes:
    match = re.compile(fr"^.*?{regex_str}.*?").match(remark)   
    if match is not None: return True #match.group(1), 0

  return False

def extract_pure_nums_from_expr(s):
  s = re.sub(',', '', s)   # 1,200 -> 1200
  s = re.sub(r'\D', ' ', s)  # 1200+sqft -> 1200,   500+1200 sqft -> [500, 1200]

  try:
    return [int(e) for e in s.split()]
  except:
    return None

def extract_unit_from_expr(s):
  # assume imperial unless proven otherwise

  for regex in UNIT_METRIC_REGEXES:
    if len(re.findall(fr"{regex}", s)) > 0:
      return 'unit_metric'

  return 'unit_imperial'


# Generate weak labels using regexes

def gen_weak_supervised_entities(remarks: str) -> List:  # List of entities
  ''' Generate weak labels from regexes '''
  entities = []
  for regex in all_regexes_2:
    search_from = 0
    for number_match, unit_match in re.findall(fr"{regex}", remarks):
      start_idx = remarks[search_from:].find(number_match)
      end_idx = start_idx + len(number_match)

      entities.append([start_idx + search_from, end_idx + search_from, 'SI'])   # for number match, default to SIZE_INTERNAL
      search_from = search_from + end_idx + 1

      start_idx = remarks[search_from:].find(unit_match)
      end_idx = start_idx + len(unit_match)
      entities.append([start_idx + search_from, end_idx + search_from, 'SU_I'])  # for unit match, default to imperial.
      
      search_from = search_from + end_idx + 1

  def mergeIntervals(entities):
    entities = sorted(entities, key=lambda tup: tup[0])   # sort by 'start' the lower bound.
    merged_entities = []

    for current_entity in entities:
      if not merged_entities:
        merged_entities.append(current_entity)        # the very first entity, no merging consideration is needed.
      else:
        prev_entity = merged_entities[-1]

        if current_entity[0] < prev_entity[1]:    # overlapped
          if prev_entity[2] is current_entity[2]:  # same type
            upper_bound = max(prev_entity[1], current_entity[1])
            merged_entities[-1] = [prev_entity[0], upper_bound, prev_entity[2]]   # simple merge by union
          else:
            if prev_entity[1] >= current_entity[1]:   # if current entity interval is a subset of previous entity's, then drop current entity
              merged_entities[-1] = prev_entity
            else:
              merged_entities[-1] = [prev_entity[0], current_entity[1], current_entity[2]]  # union and adapt the current entity type (this rule somewhat arbitrary)
        else:
            merged_entities.append(current_entity)
    return merged_entities

  entities = mergeIntervals(entities)   # cleanup overlap or duplicates
  return entities

# Generate "weak" labels using a pretrained model
def gen_weak_supervised_entities_from_model(model, tokenizer, sentence, tags):

  def merge_model_entities(entities):
    # for entities suggested by model, only merge if entity name are same type

    entities = sorted(entities, key=lambda tup: tup[0])   # sort by 'start' the lower bound.
    merged_entities = []

    for current_entity in entities:
      if not merged_entities:
        merged_entities.append(current_entity)        # the very first entity, no merging consideration is needed.
      else:
        prev_entity = merged_entities[-1]

        # consider merging if 2 similar entities are next to each other, or with a single space inbetween.
        if current_entity[0] <= prev_entity[1]+1 and prev_entity[2] == current_entity[2]:
          upper_bound = max(prev_entity[1], current_entity[1])
          merged_entities[-1] = [prev_entity[0], upper_bound, prev_entity[2]]   # simple merge by union  
        else:
          merged_entities.append(current_entity)

    return merged_entities  

  # TODO: not handling non-ascii for now, this won't work for french 
  sentence = sentence.encode('ascii', 'replace').decode('utf-8')
  
  # tokenized_sentence = tokenizer([sentence])
  tokenized_sentence = tokenizer([sentence], truncation=True, max_length=512)
  
  word_ids = tokenized_sentence.word_ids(0)
  input_ids = tokenized_sentence['input_ids']
  tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
  p = tf.nn.softmax(model.predict(input_ids).logits, axis=-1)
  predictions = np.argmax(p, axis=-1)[0]
  tag_names = [tags[i] for i in predictions]

  predictions = list(predictions[1:-1])
  word_ids = word_ids[1:-1]
  tokens = tokens[1:-1]
  tag_names = tag_names[1:-1]

  # print(tabulate([tokens, predictions, tag_names, word_ids], tablefmt='plain'))

  entities = []
  prev_word_id = None
  search_from = 0
  sentence = sentence.lower()
   
  for t, word_id, tag in zip(tokens, word_ids, tag_names):
    if prev_word_id != word_id:
      len_t = len(t)
      start_idx = sentence[search_from:].find(t)
      end_idx = start_idx + len_t
    else:
      len_t = len(t) - 2
      start_idx = sentence[search_from:].find(t[2:])     # strip 
      end_idx = start_idx + len_t

    entities.append([start_idx + search_from, end_idx + search_from, tag, True])   # last elem is a mask, will set to False if merged
    search_from = search_from + end_idx

    prev_word_id = word_id
    # break

  # for start, end, tag, mask in entities:
  #   print(sentence[start: end]) #, tag, mask)

  entities = list(reversed(entities))
  predictions = list(reversed(predictions))
  word_ids = list(reversed(word_ids))

  # merge if word_id and predictions are the same 

  previous_word_id, previous_p = None, None
  for k, (word_id, p) in enumerate(zip(word_ids, predictions)):
    if word_id == previous_word_id and p == previous_p:     # merge entities
      entities[k-1][-1] = False
      entities[k][1] = entities[k-1][1]
    previous_word_id = word_id
    previous_p = p

  entities = list(reversed(entities))
  entities = [e[:-1]for e in entities if e[-1]]

  # remove O, the empty tag  
  entities = [e for e in entities if e[-1] != 'O']

  entities = merge_model_entities(entities)
  
  return entities



if __name__ == '__main__':
  # unit tests

  def unit_test_has_number_followed_by_unit(test_strings):
    for s in test_strings:
      # print(has_number_followed_by_unit(s))
      if not has_number_followed_by_unit(s):
        print(f'Unmatched:\t{s}')      

  def unit_test_get_html_w_number_or_sq_ft_colored(test_strings):
    for s in test_strings:
      # display(HTML(get_html_w_number_or_sq_ft_colored(s, excl_lone_num=True)))
      return get_html_w_number_or_sq_ft_colored(s, excl_lone_num=True)

  def unit_test_gen_weak_supervised_entities(test_strings):
    for s in test_strings:
      print(gen_weak_supervised_entities(s))

  unit_test_has_number_followed_by_unit(test_strings)
  unit_test_get_html_w_number_or_sq_ft_colored(test_strings)
  unit_test_gen_weak_supervised_entities(test_strings)

  #  Test findall with all_regexes on test_strings
  '''
  matches = []
  for k, s in enumerate(test_strings):
    results = {'string': s, 'values': [], 'unit': [], 'matches': []}
    # parse number
    for regex in all_regexes:
      for match in re.findall(fr"{regex}", s):
        results['values'] += extract_pure_nums_from_expr(match)
        results['unit'].append(extract_unit_from_expr(match))
        results['matches'].append(match)
      
    matches.append(results)

  matches
  # '''
