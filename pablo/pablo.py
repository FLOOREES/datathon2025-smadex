import dask.dataframe as dd
import dask_ml.preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import json

# --- 1. Configuration & Column Definitions ---

# --- Paths & Training ---
TRAIN_DATA_PATH = 'data/smadex-challenge-predict-the-revenue/train/train/'
EPOCHS = 2
BATCH_SIZE = 1024
LEARNING_RATE = 0.001

# --- Target Column ---
# Let's predict a regression target, e.g., 'iap_revenue_d7'
TARGET_COLUMN = 'iap_revenue_d7' 

# --- Feature Column Subsets (Demonstrating Each Type) ---
# We'll use these lists to apply different transformations.
# You can expand these lists with all your features.

# 1. Categorical: Will be integer-encoded and fed into Embedding layers
CAT_COLS = [
    'advertiser_category',
    'country', 
    'dev_make', 
    'dev_os'
]

# 2. Numerical: Will be log-transformed (if skewed) and scaled
# Note: 'release_msrp' is a raw feature. Others will be created by us.
NUM_COLS_RAW = [
    'release_msrp'
]

# 3. Cyclical: Will be transformed with sin/cos
CYCLICAL_COLS = [
    'hour', 
    'weekday'
]

# 4. Timestamp: Will be converted to "time since" (delta) features
TIMESTAMP_COLS = [
    'release_date', # Special case (Y-M format)
    'last_buy',
    'last_ins'
]

# 5. Map/Dict: Will be unrolled or aggregated
MAP_COLS = {
    'cpm': ['b', 'i', 'r'], # Unroll 'banner', 'interstitial', 'rewarded'
    'iap_revenue_usd_bundle': 'sum' # Aggregate: get total revenue
}

# 6. List: Will be aggregated
LIST_COLS = {
    'user_bundles': 'count', # Get len() of the list
    'new_bundles': 'count'
}

# 7. Histogram: Will be aggregated
HIST_COLS = {
    'city_hist': 'total_requests' # Sum all values in the hist
}

# --- Preprocessing Parameters ---
RARE_THRESHOLD = 10 # Group categories with < 10 occurrences into '<RARE>'
EMBEDDING_DIMS = {
    'advertiser_category': 10,
    'country': 16,
    'dev_make': 24,
    'dev_os': 8,
}

# --- Utility Columns (to be dropped before training) ---
METADATA_COLS = ['row_id', 'datetime']

# Suppress Dask warnings
warnings.filterwarnings("ignore", category=UserWarning, module="dask")
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 2. The Deep Learning Model ---

class WideAndDeepModel(nn.Module):
    """
    A model that handles both categorical (via Embeddings) and 
    numerical features.
    """
    def __init__(self, vocab_sizes, embedding_dims, num_numerical_features):
        super(WideAndDeepModel, self).__init__()
        
        # --- Embedding Layers (Categorical) ---
        self.embedding_layers = nn.ModuleDict()
        total_embedding_dim = 0
        for col, vocab_size in vocab_sizes.items():
            dim = embedding_dims.get(col, 10) # Default to dim=10 if not specified
            self.embedding_layers[col] = nn.Embedding(vocab_size, dim)
            total_embedding_dim += dim
            
        print(f"Total embedding dimension: {total_embedding_dim}")
        
        # --- Deep Tower (Numerical + Embeddings) ---
        self.num_numerical_features = num_numerical_features
        input_dim = total_embedding_dim + self.num_numerical_features
        
        print(f"Total model input dim (Embeds + Numericals): {input_dim}")
        
        self.tower = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # Output 1 value (target)
        )

    def forward(self, x_cat, x_num):
        """
        x_cat: A dictionary of {feature_name: tensor}
        x_num: A single tensor of numerical features
        """
        # --- Process Embeddings ---
        embeddings = []
        for col, tensor in x_cat.items():
            embeddings.append(self.embedding_layers[col](tensor))
        
        # Concatenate all embedding outputs
        x_embed = torch.cat(embeddings, dim=1)
        
        # --- Concatenate Numerical ---
        x_combined = torch.cat([x_embed, x_num], dim=1)
        
        # --- Pass through Tower ---
        return self.tower(x_combined)


# --- 3. Custom PyTorch Dataset ---

class HeterogeneousDataset(Dataset):
    """
    A Dataset to handle our two types of inputs:
    1. A dictionary of categorical tensors
    2. A single tensor of numerical features
    """
    def __init__(self, cat_data, num_data, target_data):
        self.cat_data = {col: torch.tensor(data, dtype=torch.long) for col, data in cat_data.items()}
        self.num_data = torch.tensor(num_data, dtype=torch.float32)
        self.target_data = torch.tensor(target_data, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        # All inputs must have the same length
        return len(self.target_data)

    def __getitem__(self, idx):
        # Get categorical features for this index
        x_cat = {col: data[idx] for col, data in self.cat_data.items()}
        
        # Get numerical features for this index
        x_num = self.num_data[idx]
        
        # Get target for this index
        y = self.target_data[idx]
        
        return (x_cat, x_num), y


# --- 4. Feature Engineering & Preprocessing Class ---

class FeatureProcessor:
    """
    Handles fitting vocabs/scalers on Dask and transforming
    Pandas partitions during training.
    """
    def __init__(self, cat_cols, num_cols_raw, cyclical_cols, ts_cols, map_cols, list_cols, hist_cols, target_col):
        # Store column names
        self.CAT_COLS = cat_cols
        self.NUM_COLS_RAW = num_cols_raw
        self.CYCLICAL_COLS = cyclical_cols
        self.TS_COLS = ts_cols
        self.MAP_COLS = map_cols
        self.LIST_COLS = list_cols
        self.HIST_COLS = hist_cols
        self.TARGET_COL = target_col

        # These will be "fit"
        self.vocabularies = {}
        self.vocab_sizes = {}
        self.scaler = StandardScaler() # Use sklearn scaler, fit on 1st partition
        self.engineered_num_cols = []
        self._is_scaler_fit = False

    
    def fit_vocabs(self, ddf, rare_threshold):
        """
        Fits vocabularies on the full Dask dataframe.
        This is a global operation and should be done once.
        """
        print("Computing vocabularies...")
        for col in self.CAT_COLS:
            print(f"  Fitting vocab for: {col}")
            # Get value counts, compute, and filter
            counts = ddf[col].value_counts().compute()
            frequent_labels = counts[counts >= rare_threshold].index.tolist()
            
            # Create vocab with special tokens
            vocab = defaultdict(lambda: 1) # 1 is '<RARE>'
            vocab['<MISSING>'] = 0
            vocab['<RARE>'] = 1
            for i, label in enumerate(frequent_labels, 2):
                vocab[label] = i
                
            self.vocabularies[col] = vocab
            self.vocab_sizes[col] = len(vocab)
            print(f"    Vocab size for {col}: {len(vocab)}")
    
    
    def engineer_features_partition(self, pdf, ref_datetime_col):
        """
        Applies all feature engineering logic to a single Pandas partition.
        This function CREATES the new numerical features.
        """
        
        # Create a copy to avoid SettingWithCopyWarning
        pdf_out = pd.DataFrame(index=pdf.index)
        ref_datetime = pd.to_datetime(ref_datetime_col)

        # --- 1. Pass-through Raw Features ---
        pdf_out[self.CAT_COLS] = pdf[self.CAT_COLS].fillna('<MISSING>')
        pdf_out[self.NUM_COLS_RAW] = pdf[self.NUM_COLS_RAW]
        pdf_out[self.TARGET_COL] = pdf[self.TARGET_COL]

        # --- 2. Cyclical Features ---
        for col in self.CYCLICAL_COLS:
            if col == 'hour':
                max_val = 23
            else: # weekday
                max_val = 6
            pdf_out[f'{col}_sin'] = np.sin(2 * np.pi * pdf[col] / max_val)
            pdf_out[f'{col}_cos'] = np.cos(2 * np.pi * pdf[col] / max_val)

        # --- 3. Timestamp (Delta) Features ---
        for col in self.TS_COLS:
            if col == 'release_date':
                # Special parser for "2023_october"
                ts = pd.to_datetime(pdf[col].str.replace('_', ' '), format='%Y %B', errors='coerce')
            else:
                # Standard unix timestamps
                ts = pd.to_datetime(pdf[col], unit='s', errors='coerce')
            
            # Calculate delta in days
            delta_days = (ref_datetime - ts).dt.total_seconds() / (60 * 60 * 24)
            pdf_out[f'days_since_{col}'] = delta_days

        # --- 4. Map/Dict Features ---
        for col, action in self.MAP_COLS.items():
            if isinstance(action, list): # Unroll
                # Handle missing/empty maps
                def safe_parse_map(x):
                    if isinstance(x, list): # Handle '[(k, v)]' format
                        return {k: v for k, v in x}
                    if isinstance(x, dict):
                        return x
                    return {}
                
                parsed_maps = pdf[col].apply(safe_parse_map)
                for key in action:
                    pdf_out[f'{col}_{key}'] = parsed_maps.apply(lambda m: m.get(key))
            
            elif action == 'sum': # Aggregate
                def safe_sum_map_values(x):
                    try:
                        if isinstance(x, list): # '[(k, v)]'
                            return sum(v for k, v in x)
                        if isinstance(x, dict):
                            return sum(x.values())
                    except:
                        return 0
                    return 0
                pdf_out[f'{col}_sum'] = pdf[col].apply(safe_sum_map_values)

        # --- 5. List Features ---
        for col, action in self.LIST_COLS.items():
            if action == 'count':
                pdf_out[f'{col}_count'] = pdf[col].apply(lambda x: len(x) if isinstance(x, list) else 0)

        # --- 6. Histogram Features ---
        for col, action in self.HIST_COLS.items():
            if action == 'total_requests':
                # Hist format is '[(key, count)]'
                def safe_sum_hist(x):
                    try:
                        return sum(count for key, count in x)
                    except:
                        return 0
                pdf_out[f'{col}_total_requests'] = pdf[col].apply(safe_sum_hist)
        
        # --- Store list of all engineered numerical columns ---
        if not self.engineered_num_cols:
            all_cols = set(pdf_out.columns)
            cat_target = set(self.CAT_COLS) | set([self.TARGET_COL])
            self.engineered_num_cols = sorted(list(all_cols - cat_target))
            print(f"Discovered {len(self.engineered_num_cols)} engineered numerical features.")
            print(f"First 5: {self.engineered_num_cols[:5]}")
        
        return pdf_out[self.CAT_COLS + self.engineered_num_cols + [self.TARGET_COL]]

    
    def transform_partition(self, pdf, ref_datetime_col):
        """
        Applies all transforms to a partition and returns data
        ready for the model.
        """
        # 1. Engineer all features
        pdf_fe = self.engineer_features_partition(pdf, ref_datetime_col)
        
        # 2. Separate data
        pdf_cat = pdf_fe[self.CAT_COLS]
        pdf_num = pdf_fe[self.engineered_num_cols]
        pdf_target = pdf_fe[self.TARGET_COL]

        # 3. Apply Vocabularies (Categorical)
        cat_data = {}
        for col in self.CAT_COLS:
            vocab = self.vocabularies[col]
            # Map values, fill NaNs from mapping with '<MISSING>' token (0)
            cat_data[col] = pdf_cat[col].map(vocab).fillna(0).values
        
        # 4. Apply Scaling (Numerical)
        # Handle NAs/Infs from log/deltas before scaling
        pdf_num = pdf_num.fillna(0).replace([np.inf, -np.inf], 0)
        
        if not self._is_scaler_fit:
            print("Fitting StandardScaler on first partition...")
            self.scaler.fit(pdf_num)
            self._is_scaler_fit = True
        
        num_data = self.scaler.transform(pdf_num)
        
        # 5. Get Target
        target_data = pdf_target.fillna(0).values
        
        return cat_data, num_data, target_data


# --- 5. Main Training Function ---

def main():
    """
    Main function to fit preprocessors and train the model.
    """
    
    # --- Setup Device (CUDA or CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Dask DataFrame (Lazily) ---
    print(f"Loading data from: {TRAIN_DATA_PATH}")
    
    try:
        # Define dtypes for complex columns to help Dask
        # This is often necessary for lists/dicts
        dtypes = {
            'bcat': 'object',
            'bcat_bottom_taxonomy': 'object',
            'bundles_cat': 'object',
            'bundles_cat_bottom_taxonomy': 'object',
            'bundles_ins': 'object',
            'city_hist': 'object',
            'country_hist': 'object',
            'cpm': 'object',
            'cpm_pct_rk': 'object',
            'ctr': 'object',
            'ctr_pct_rk': 'object',
            'dev_language_hist': 'object',
            'dev_osv_hist': 'object',
            'first_request_ts_bundle': 'object',
            'first_request_ts_category_bottom_taxonomy': 'object',
            'hour_ratio': 'object',
            'iap_revenue_usd_bundle': 'object',
            'iap_revenue_usd_category': 'object',
            'iap_revenue_usd_category_bottom_taxonomy': 'object',
            'last_buy_ts_bundle': 'object',
            'last_buy_ts_category': 'object',
            'last_install_ts_bundle': 'object',
            'last_install_ts_category': 'object',
            'advertiser_actions_action_count': 'object',
            'advertiser_actions_action_last_timestamp': 'object',
            'user_actions_bundles_action_count': 'object',
            'user_actions_bundles_action_last_timestamp': 'object',
            'new_bundles': 'object',
            'num_buys_bundle': 'object',
            'num_buys_category': 'object',
            'num_buys_category_bottom_taxonomy': 'object',
            'region_hist': 'object',
            'rev_by_adv': 'object',
            'rwd_prank': 'object',
            'user_bundles': 'object',
            'user_bundles_l28d': 'object',
            'whale_users_bundle_num_buys_prank': 'object',
            'whale_users_bundle_revenue_prank': 'object',
            'whale_users_bundle_total_num_buys': 'object',
            'whale_users_bundle_total_revenue': 'object'
        }
        
        all_cols = (
            CAT_COLS + NUM_COLS_RAW + CYCLICAL_COLS + TIMESTAMP_COLS + 
            list(MAP_COLS.keys()) + list(LIST_COLS.keys()) + 
            list(HIST_COLS.keys()) + [TARGET_COLUMN] + METADATA_COLS
        )
        
        # Read only the columns we need
        ddf = dd.read_parquet(
            TRAIN_DATA_PATH, 
            engine='pyarrow', 
            columns=list(set(all_cols)) # Use set for uniqueness
        )
        
        # Ensure 'datetime' is parsed as a Dask datetime series
        ddf['datetime'] = dd.to_datetime(ddf['datetime'])

    except Exception as e:
        print(f"Error reading parquet metadata from {TRAIN_DATA_PATH}. {e}")
        return

    # --- 1. FIT PREPROCESSORS ---
    print("\n--- Starting Preprocessing 'fit' Step ---")
    processor = FeatureProcessor(
        CAT_COLS, NUM_COLS_RAW, CYCLICAL_COLS, TIMESTAMP_COLS,
        MAP_COLS, LIST_COLS, HIST_COLS, TARGET_COLUMN
    )
    
    # Fit vocabularies (requires a pass over the data)
    processor.fit_vocabs(ddf, rare_threshold=RARE_THRESHOLD)
    
    # NOTE: We will fit the scaler on the *first partition* during training.
    # Fitting a dask_ml.StandardScaler requires engineering all features
    # on the *entire* dask dataframe first, which is another full pass.
    # For speed in this example, we'll fit on the first batch.
    # For a production model, you should fit on the full ddf.
    print("--- Preprocessing 'fit' Complete ---")
    

    # --- 2. INITIALIZE MODEL ---
    # We must wait until after fitting vocabs to know the vocab sizes
    
    # We also need to know the *final* number of numerical features.
    # We'll get this by processing a dummy partition (the metadata).
    print("Discovering engineered feature dimensions...")
    dummy_pdf = ddf._meta_nonempty.copy()
    dummy_dt = pd.Series([pd.Timestamp.now()] * len(dummy_pdf), index=dummy_pdf.index)
    _ = processor.engineer_features_partition(dummy_pdf, dummy_dt)
    
    num_numerical_features = len(processor.engineered_num_cols)
    
    model = WideAndDeepModel(
        vocab_sizes=processor.vocab_sizes,
        embedding_dims=EMBEDDING_DIMS,
        num_numerical_features=num_numerical_features
    ).to(device)
    
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train() # Set model to training mode

    # --- 3. TRAINING LOOP: PARTITION BY PARTITION ---
    for epoch in range(EPOCHS):
        print(f"\n--- Starting Epoch {epoch + 1}/{EPOCHS} ---")
        
        # Get a list of delayed tasks, one for each partition file
        partition_iterator = ddf.to_delayed()
        
        for i, partition in enumerate(partition_iterator):
            # 1. Load *only this single partition* into memory as a Pandas DF
            pdf = partition.compute()
            
            if pdf.empty:
                print(f"  Skipping empty partition {i+1}")
                continue
                
            # 2. Extract reference datetime and drop metadata
            ref_datetime_col = pdf['datetime']
            pdf_features = pdf.drop(columns=METADATA_COLS, errors='ignore')

            # 3. Apply all preprocessing and transformations
            cat_data, num_data, target_data = processor.transform_partition(
                pdf_features, 
                ref_datetime_col
            )
            
            # 4. Create custom Dataset and DataLoader
            partition_dataset = HeterogeneousDataset(cat_data, num_data, target_data)
            partition_loader = DataLoader(
                dataset=partition_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=True
            )

            # 5. Mini-Batch Training Loop (for this partition)
            for batch_num, ((batch_cat, batch_num_features), batch_target) in enumerate(partition_loader):
                
                # Move data to GPU
                batch_cat = {k: v.to(device) for k, v in batch_cat.items()}
                batch_num_features = batch_num_features.to(device)
                batch_target = batch_target.to(device)

                # Forward pass
                outputs = model(batch_cat, batch_num_features)
                loss = criterion(outputs, batch_target)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"  Epoch {epoch+1} | Processed Partition {i+1} | Last Batch Loss: {loss.item():.4f}")

    print("\n--- Training Finished ---")


if __name__ == "__main__":
    main()