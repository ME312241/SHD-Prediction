import pandas as pd
from tqdm import tqdm
import os

def csv_to_parquet(csv_file, parquet_file=None, chunksize=200_000):
    if parquet_file is None:
        parquet_file = csv_file.replace('.csv', '.parquet')
    
    print(f"Converting:\n   {csv_file}\n→  {parquet_file}")
    
    # Get total rows for progress bar (optional but nice)
    total_rows = sum(1 for _ in open(csv_file, 'r', encoding='utf-8', errors='ignore')) - 1
    print(f"Total rows to convert: {total_rows:,}\n")
    
    first_chunk = True
    rows_processed = 0
    
    for chunk in tqdm(pd.read_csv(csv_file, chunksize=chunksize, dtype=str, low_memory=True),
                      total=total_rows//chunksize + 1, desc="Converting to Parquet"):
        
        # Convert string columns to proper numeric types where possible
        for col in chunk.columns:
            if col.startswith(('out.', 'in.sqft', 'in.number_of_stories', 'in.aspect_ratio', 
                             'in.rotation', 'in.airtightness', 'in.tstat', 'in.weekday', 
                             'in.weekend')):
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        
        if first_chunk:
            chunk.to_parquet(parquet_file, index=False, compression='snappy')
            first_chunk = False
        else:
            chunk.to_parquet(parquet_file, index=False, compression='snappy', append=True)
        
        rows_processed += len(chunk)
    
    final_size = os.path.getsize(parquet_file) / (1024**3)  # in GB
    print(f"\n✅ Conversion complete!")
    print(f"   Parquet file size: {final_size:.2f} GB")
    print(f"   Rows written: {rows_processed:,}")


if __name__ == "__main__":
    csv_to_parquet("combined_comstock_full.csv")