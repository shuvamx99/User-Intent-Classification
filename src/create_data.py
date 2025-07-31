from faker import Faker
from datetime import datetime
import pandas as pd
import random

# Initialize Faker
fake = Faker()

# Number of records to generate
num_records = 500

# Generate data
data = {
    'user': [f'user_{random.randint(1, 490)}' for _ in range(num_records)],
    'country': [fake.country_code() for _ in range(num_records)],
    'ip_address': [fake.ipv4() for _ in range(num_records)],
    'time': [fake.date_time_this_year() for _ in range(num_records)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV (optional)
df.to_csv('../data/generated_data.csv', index=False)

# Print first few rows
print(df.head())