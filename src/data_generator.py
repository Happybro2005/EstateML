import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config
from utils import logger

def generate_dob(min_age, max_age):
    """Generates a random date of birth string based on age range."""
    today = datetime.now()
    start_date = today - timedelta(days=max_age * 365)
    end_date = today - timedelta(days=min_age * 365)
    random_days = random.randint(0, (end_date - start_date).days)
    dob = start_date + timedelta(days=random_days)
    return dob.strftime("%Y-%m-%d")

def generate_datasets(num_clients=1000, seed=42, write_to_disk=True, clients_path=None, properties_path=None):
    """
    Generates synthetic client and property datasets matching the actual test dataset schema,
    complete with missing values and duplicates for cleaning validation.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    logger.info(f"Generating mock datasets: {num_clients} clients...")
    
    client_records = []
    property_records = []
    
    # Target profile ratios: First-Time (35%), Global (25%), Corporate (20%), Luxury (20%)
    profiles = ["First-Time", "Global", "Corporate", "Luxury"]
    profile_choices = np.random.choice(profiles, size=num_clients, p=[0.35, 0.25, 0.20, 0.20])
    
    countries_pool = ["USA", "Canada", "Germany", "Belgium", "Mexico", "Russia", "UK", "Denmark", "France", "Australia"]
    regions_pool = ["California", "England", "Berlin", "Quebec", "Brussels"]
    referrals_pool = ["Website", "Agency", "Client"]
    
    for i in range(num_clients):
        client_id = f"C{i+1:04d}"
        profile = profile_choices[i]
        
        # Default initialization
        client_type = "Individual"
        gender = random.choice(["F", "M"])
        country = "USA"
        region = random.choice(regions_pool)
        dob = ""
        acquisition_purpose = "Home"
        loan_applied = "No"
        referral_channel = "Website"
        satisfaction_score = 4.0
        
        # Property initialization
        listing_id = i + 1001
        tower_number = random.randint(1, 10)
        unit_number = random.randint(1, 100)
        unit_category = "Apartment"
        sale_price = 250000
        floor_area_sqft = 800
        listing_status = "Sold"
        client_ref = client_id
        
        if profile == "First-Time":
            client_type = "Individual"
            country = "UK"
            region = "England"
            dob = generate_dob(24, 34)
            acquisition_purpose = "Home"
            loan_applied = "Yes" if random.random() < 0.88 else "No"
            referral_channel = random.choice(["Website", "Agency"])
            satisfaction_score = int(np.clip(np.random.normal(3.8, 0.8), 1, 5))
            
            unit_category = "Apartment"
            sale_price = int(random.uniform(130000, 360000))
            floor_area_sqft = int(random.uniform(450, 1150))
            
        elif profile == "Global":
            client_type = "Individual"
            country = random.choice(["Germany", "Belgium", "France"])
            region = "Berlin"
            dob = generate_dob(40, 60)
            acquisition_purpose = "Investment"
            loan_applied = "No" if random.random() < 0.94 else "Yes"
            referral_channel = random.choice(["Agency", "Website"])
            satisfaction_score = int(np.clip(np.random.normal(4.2, 0.6), 1, 5))
            
            unit_category = random.choice(["Apartment", "Apartment", "Office"])
            sale_price = int(random.uniform(400000, 600000))
            floor_area_sqft = int(random.uniform(850, 1500))
            
        elif profile == "Corporate":
            client_type = "Company"
            country = "USA"
            region = "California"
            dob = generate_dob(35, 55)
            acquisition_purpose = "Investment"
            loan_applied = "Yes" if random.random() < 0.45 else "No"
            referral_channel = "Agency"
            satisfaction_score = int(np.clip(np.random.normal(3.5, 0.9), 1, 5))
            
            unit_category = "Office"
            sale_price = int(random.uniform(500000, 700000))
            floor_area_sqft = int(random.uniform(1200, 1900))
            
        elif profile == "Luxury":
            client_type = "Individual"
            country = random.choice(["USA", "Canada"])
            region = "Quebec"
            dob = generate_dob(45, 72)
            acquisition_purpose = "Home"
            loan_applied = "No" if random.random() < 0.96 else "Yes"
            referral_channel = "Client"
            satisfaction_score = int(np.clip(np.random.normal(4.6, 0.5), 1, 5))
            
            unit_category = "Apartment"
            sale_price = int(random.uniform(600000, 730000))
            floor_area_sqft = int(random.uniform(1500, 1950))
            
        # Append client record
        client_records.append({
            config.COL_CLIENT_ID: client_id,
            config.COL_CLIENT_TYPE: client_type,
            config.COL_GENDER: gender,
            config.COL_COUNTRY: country,
            config.COL_REGION: region,
            config.COL_DOB: dob,
            config.COL_ACQUISITION_PURPOSE: acquisition_purpose,
            config.COL_LOAN_APPLIED: loan_applied,
            config.COL_REFERRAL_CHANNEL: referral_channel,
            config.COL_SATISFACTION: satisfaction_score
        })
        
        # Append property record
        property_records.append({
            config.COL_PROPERTY_ID: listing_id,
            "tower_number": tower_number,
            "transaction_date": "01-01-2024",
            config.COL_PROPERTY_TYPE: unit_category,
            "unit_number": unit_number,
            config.COL_SIZE: floor_area_sqft,
            config.COL_PURCHASE_PRICE: sale_price,
            "listing_status": listing_status,
            "client_ref": client_ref
        })
        
    df_clients = pd.DataFrame(client_records)
    df_properties = pd.DataFrame(property_records)
    
    # Inject data issues (Missing values and duplicates)
    num_client_dupes = int(num_clients * 0.02)
    dupe_indices = np.random.choice(range(num_clients), size=num_client_dupes, replace=False)
    dupe_clients = df_clients.iloc[dupe_indices].copy()
    dupe_properties = df_properties.iloc[dupe_indices].copy()
    
    df_clients = pd.concat([df_clients, dupe_clients], ignore_index=True)
    df_properties = pd.concat([df_properties, dupe_properties], ignore_index=True)
    
    # Missing values injection
    for col, prob in [(config.COL_CLIENT_TYPE, 0.01), (config.COL_GENDER, 0.02), 
                      (config.COL_COUNTRY, 0.015), (config.COL_REFERRAL_CHANNEL, 0.02),
                      (config.COL_SATISFACTION, 0.03)]:
        mask = np.random.random(size=len(df_clients)) < prob
        df_clients.loc[mask, col] = np.nan
        
    for col, prob in [(config.COL_PURCHASE_PRICE, 0.01), (config.COL_SIZE, 0.01)]:
        mask = np.random.random(size=len(df_properties)) < prob
        df_properties.loc[mask, col] = np.nan
        
    if write_to_disk:
        c_path = clients_path if clients_path else config.RAW_CLIENTS_PATH
        p_path = properties_path if properties_path else config.RAW_PROPERTIES_PATH
        
        os.makedirs(os.path.dirname(c_path), exist_ok=True)
        df_clients.to_csv(c_path, index=False)
        df_properties.to_csv(p_path, index=False)
        
        logger.info(f"Successfully generated datasets and saved to disk.")
        logger.info(f"Clients dataset size: {df_clients.shape}. Saved to {c_path}")
        logger.info(f"Properties dataset size: {df_properties.shape}. Saved to {p_path}")
    else:
        logger.info(f"Successfully generated datasets in-memory (write_to_disk=False).")
        
    return df_clients, df_properties

if __name__ == "__main__":
    generate_datasets()
