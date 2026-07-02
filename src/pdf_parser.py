import os
import pypdf
import pandas as pd
import numpy as np

def parse_properties_pdf():
    # Use dynamic and relative paths for cross-platform support
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(os.path.expanduser("~"), "Downloads", "properties.pdf")
    csv_output_path = os.path.join(base_dir, "data", "properties.csv")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
    print(f"Loading PDF from {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    records = []
    
    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}. Starting parsing...")
    
    for idx in range(0, total_pages, 2):
        left_lines = reader.pages[idx].extract_text().split('\n')
        right_lines = reader.pages[idx+1].extract_text().split('\n')
        
        left_rows = []
        for line in left_lines:
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                left_rows.append(parts)
                
        right_rows = []
        for line in right_lines:
            line = line.strip()
            if line.startswith('$'):
                right_rows.append(line)
                
        if len(left_rows) != len(right_rows):
            print(f"Warning: mismatch at page pair {idx}-{idx+1}: Left={len(left_rows)} | Right={len(right_rows)}")
            
        for left, right in zip(left_rows, right_rows):
            listing_id = int(left[0])
            tower_number = int(left[1])
            
            # Date + Category formatting (Date is 10 chars DD-MM-YYYY)
            date_cat = left[2]
            transaction_date = date_cat[:10]
            unit_category = date_cat[10:]
            
            unit_number = int(left[3])
            floor_area_sqft = float(left[4])
            
            if "Available" in right:
                listing_status = "Available"
                client_ref = ""
                price_str = right.replace("Available", "").strip()
            else:
                parts_r = right.split()
                client_ref = parts_r[-1]
                status_price = parts_r[0]
                if status_price.endswith("Sold"):
                    listing_status = "Sold"
                    price_str = status_price[:-4]
                else:
                    listing_status = "Sold"
                    price_str = status_price
            
            price_val = float(price_str.replace("$", "").replace(",", ""))
            
            records.append({
                "listing_id": listing_id,
                "tower_number": tower_number,
                "transaction_date": transaction_date,
                "unit_category": unit_category,
                "unit_number": unit_number,
                "floor_area_sqft": floor_area_sqft,
                "sale_price": price_val,
                "listing_status": listing_status,
                "client_ref": client_ref
            })
            
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df.to_csv(csv_output_path, index=False)
    print(f"Successfully wrote {len(df)} records to {csv_output_path}")

if __name__ == "__main__":
    parse_properties_pdf()
