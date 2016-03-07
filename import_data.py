#import_data.py

# SECTION : LOAD DATA
import pandas as pd
import os
import re
import numpy as np

# Fettermania libraries
import clean_text


def create_query_datasets_crowdflower(small=False):
  df = pd.read_csv('./data/crowdflower.csv').dropna().reset_index()
  if small:
    df = df[:100]  
  df['original_text'] = df['product_title'] + " " + df['product_description']
  df = import_preprocess(df)
  #df['query'] is already in there
  # Fettermania TODO: not strictly classification.  Can be 0-4.
  df['relevant'] = df['median_relevance'].map(lambda x: 1 if x == 4 else 0)
  del df['product_description']
  del df['median_relevance']
  del df['relevance_variance']
  
  normalization_sample_ct = round(df.shape[0] / 3)
  df_normalization_sample = df[-normalization_sample_ct:]
  df = df[0:-normalization_sample_ct]
  df['doc_index'] = np.array(range(df.shape[0]))  # Fettermania TODO: This is not random-ish.

  return (df_normalization_sample['cleaned_text'], df[['cleaned_text', 'original_text', 'doc_index']], df)


 

def import_data_frame_toy():
  df = pd.DataFrame(np.array([
    'Cats are similar in anatomy to the other felids, with a strong, flexible body, quick reflexes, sharp retractable claws, and teeth adapted to killing small prey. Cat senses fit a crepuscular and predatory ecological niche. Cats can hear sounds too faint or too high in frequency for human ears, such as those made by mice and other small animals. They can see in near darkness. Like most other mammals, cats have poorer color vision and a better sense of smell than humans. Cats, despite being solitary hunters, are a social species and cat communication includes the use of a variety of vocalizations (mewing, purring, trilling, hissing, growling, and grunting), as well as cat pheromones and types of cat-specific body language.[8]',
    'Dogs love to eat and run around.',
    'It was raining cats and dogs the other night... so bad that I couldn\'t go outside. Sometimes I would come to the window and just stare at the rain. It was very depressing, but in the morning, I felt better!',
    'It\'s a dog-eat-dog world out there. From puppies to big hounds, everyone struggles to survive, to avoid his superior and to beat up on his inferior. That\'s just how it is.',
    'Cats are cool, soft, fuzzy and bouncy!',
    'Cats and dogs are two common types of household animals. There are many species of cats and dogs - from the common house cat, to the Blue Russian, from bulldog to shepherd. Both cats and dogs have been domesticated by man many thousands of years ago and are loved and cared for by many pet owners today. There are even urban legends of cat owners having statistically better health than non-cat owners - and everyone knows how useful a dog can be, for protecting the house, for instance! There are many more things to say about cats and dogs, but I think I\'ve run out of time, so I have to go. Thank you for listening!',
    'The domesticated cat (Latin: Felis catus) or the undomesticated cat (Latin: Felis silvestris catus) is a small, typically furry, carnivorous mammal',
    'In comparison to dogs, cats have not undergone major changes during the domestication process, as the form and behavior of the domestic cat is not radically different from those of wildcats and domestic cats are perfectly capable of surviving in the wild',
    'Cats, like dogs, are digitigrades',
    'Cats do eat grass occasionally.',
    'Cats can hear higher-pitched sounds than either dogs or humans, detecting frequencies from 55 Hz to 79,000 Hz, a range of 10.5 octaves, while humans and dogs both have ranges of about 9 octaves.',
    'The average lifespan of pet cats has risen in recent years. In the early 1980s it was about seven years,[96]:33[97] rising to 9.4 years in 1995[96]:33 and 12–15 years in 2014.[98] However, cats have been reported as surviving into their 30s,[99] with the oldest known cat, Creme Puff, dying at a verified age of 38.[100] Spaying or neutering increases life expectancy: one study found neutered male cats live twice as long as intact males, while spayed female cats live 62% longer than intact females.[96]:35 Having a cat neutered confers health benefits, because castrated males cannot develop testicular cancer, spayed females cannot develop uterine or ovarian cancer, and both have a reduced risk of mammary cancer.[101]']))
  df.columns = ["original_text"]
  return df


def import_query_data_frame_toy():
  df = pd.DataFrame(np.array([
    'cat',
    'dog',
    'cat dog'
    ]))
  df.columns = ["query"]
  return df

# Fettermania: Cartesian product - might want to reduce repeated 
# text payloads with keys instead
def create_query_datasets_toy(simulated_relevant_p=.5):
  df = import_data_frame_toy();
  df = import_preprocess(df); # cleaned_text, original_text

  normalization_sample_ct = round(df.shape[0] / 3)
  df_normalization_sample = df[-normalization_sample_ct:]
  df = df[0:-normalization_sample_ct]
  df['doc_index'] = np.array(range(df.shape[0]))  # Fettermania TODO: This is not random-ish.
  qdf = import_query_data_frame_toy();

  # Fettermania: Hack cartesian product
  df['join_key'] = np.repeat(1, df.shape[0])
  qdf['join_key'] = np.repeat(1, qdf.shape[0])
  cartesian = pd.merge(df,qdf, on='join_key')
  del df['join_key']
  del qdf['join_key']
  del cartesian['join_key']

  np.random.seed(5) # wide "looking" sample
  relevant = np.random.rand(cartesian.shape[0])
  relevant = (relevant < simulated_relevant_p) + 0
  cartesian['relevant'] = relevant
  return (df_normalization_sample['cleaned_text'], df, cartesian)


# preprocess
def import_preprocess(df):
  df['cleaned_text'] = df['original_text'].apply(clean_text.clean_text)
  return df # passed by ref, TODO fix
  