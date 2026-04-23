''' Update notes by PRK Nov 10:
        Taken out zipcode column entirely and added three more column removals (have commented in front)
        Also commented out the print order'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import datetime as dt


filename = '../Data/listings.csv'
reviews_filename = '../Data/reviews_cleaned.csv'
data = pd.read_csv(filename, low_memory=False)
reviews = pd.read_csv(reviews_filename, names=['listing_id', 'comments'])

print(len(data.columns))

# Drop unused columns
data = data.drop(columns=[
    'host_name', 'notes', 'host_about', 'calendar_updated',
    'host_acceptance_rate', 'description', 'thumbnail_url',
    'experiences_offered', 'listing_url', 'name', 'summary',
    'space', 'scrape_id', 'last_scraped', 'neighborhood_overview',
    'transit', 'access', 'interaction', 'house_rules',
    'medium_url', 'picture_url', 'xl_picture_url', 'host_url',
    'host_thumbnail_url', 'host_picture_url', 'smart_location',
    'license', 'jurisdiction_names', 'street', 'neighbourhood',
    'country', 'country_code', 'host_location', 'host_neighbourhood',
    'market', 'is_location_exact', 'square_feet', 'weekly_price',
    'monthly_price', 'availability_30', 'availability_60',
    'availability_90', 'availability_365', 'calendar_last_scraped',
    'first_review', 'last_review', 'requires_license',
    'calculated_host_listings_count', 'host_listings_count', 'zipcode'
])


print('Splitting host verifications...')
data['host_verifications'] = data['host_verifications'].fillna('').astype(str)

host_verification_set = set()
def collect_host_verifications(entry):
    entry_list = entry.replace("[","").replace("]","").replace("'","").replace('"',"").replace(" ","").split(',')
    for v in entry_list:
        if v and v != 'None':
            host_verification_set.add(v + "_verification")

data['host_verifications'].apply(collect_host_verifications)

# Batch generate columns to avoid fragmentation warnings
verif_df = pd.DataFrame(0, index=data.index, columns=list(host_verification_set))

def generic_verification(entry, v):
    entry_list = entry.replace("[","").replace("]","").replace("'","").replace('"',"").replace(" ","").split(',')
    return 1 if v.replace("_verification", "") in entry_list else 0

for v in host_verification_set:
    verif_df[v] = data['host_verifications'].apply(lambda x: generic_verification(x, v))

data = pd.concat([data, verif_df], axis=1)
data = data.drop(columns=['host_verifications'])


def clean_superhost(entry):
    return 1 if entry == 't' else 0

bool_cols = [
    'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
    'has_availability', 'instant_bookable', 'is_business_travel_ready',
    'require_guest_profile_picture', 'require_guest_phone_verification'
]
for c in bool_cols:
    data[c] = data[c].apply(clean_superhost)


data['host_response_rate'] = data['host_response_rate'].astype(str).str.replace('%', '').replace('nan', '0').astype(int)


def clean_number_removal(entry):
    return -55 if pd.isna(entry) else entry

for c in ['bathrooms', 'bedrooms', 'beds']:
    data[c] = data[c].apply(clean_number_removal)

data = data[(data['bathrooms'] != -55) & (data['bedrooms'] != -55) & (data['beds'] != -55)]
data['reviews_per_month'] = data['reviews_per_month'].fillna(0)

# Clean price columns
def clean_price(entry):
    if pd.isna(entry): return -55
    e = str(entry).replace('$', '').replace(',', '')
    return -55 if float(e) == 0 else np.log(float(e))

for c in ['price', 'extra_people', 'security_deposit', 'cleaning_fee']:
    data[c] = data[c].apply(clean_price)

data = data[data['price'] != -55]
data['host_total_listings_count'] = data['host_total_listings_count'].fillna(1)

# Filter for New York
print("Cleaning the state...")
def clean_state(e):
    if isinstance(e, str):
        return 'NY' if e.upper() in ['NY', 'NEW YORK'] else e
    return ''

data['state'] = data['state'].apply(clean_state)
data = data[data['state'] == 'NY']


print('Splitting amenities (fast mode)...')
data['amenities'] = data['amenities'].fillna('').astype(str)

amenity_set = set()
def collect_amenities(entry):
    items = entry.replace('{','').replace('}','').replace('"','').replace("'",'').replace(' ','_').split(',')
    for i in items:
        if i and 'translation_missing' not in i:
            amenity_set.add(i)

data['amenities'].apply(collect_amenities)


amenity_df = pd.DataFrame(0, index=data.index, columns=list(amenity_set))

def has_amenity(entry, a):
    return 1 if a in entry.replace('{','').replace('}','').replace('"','').replace("'",'').replace(' ','_').split(',') else 0

for a in amenity_set:
    amenity_df[a] = data['amenities'].apply(lambda x: has_amenity(x, a))

data = pd.concat([data, amenity_df], axis=1)
data = data.drop(columns=['amenities', 'state'])

# One-hot encoding for categorical features
cat_cols = ['property_type', 'bed_type', 'room_type', 'neighbourhood_group_cleansed',
            'city', 'cancellation_policy', 'host_response_time', 'neighbourhood_cleansed']
for c in cat_cols:
    if c in data.columns:
        data = pd.get_dummies(data, columns=[c], prefix=c)

# Clean host registration date
data['host_since'] = data['host_since'].fillna('')
data = data[data['host_since'] != '']
dummy = dt.datetime(2018,11,10)
data['host_since'] = (dummy - pd.to_datetime(data['host_since'])).dt.days.astype(float)

# Fill missing review scores with 0
review_cols = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
               'review_scores_checkin', 'review_scores_communication',
               'review_scores_location', 'review_scores_value']
for c in review_cols:
    data[c] = data[c].fillna(0)

# Merge review data
data = data.set_index('id').join(reviews.set_index('listing_id'))
data['comments'] = data['comments'].fillna(0)

# Save cleaned dataset
data.to_csv('../Data/data_cleaned.csv')
print("Processing completed! Saved to ../Data/data_cleaned.csv")