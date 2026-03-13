# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset

# # --- Configuration ---
# # 1. Force CPU
# DEVICE = torch.device("cpu") 

# # 2. File Paths (Raw Strings for Windows paths)
# TRAIN_PATH = r"E:\NewDownloads\cf2\ml-100k\u4.base"
# TEST_PATH  = r"E:\NewDownloads\cf2\ml-100k\u4.test"

# # 3. Hyperparameters
# BATCH_SIZE = 128
# LEARNING_RATE = 0.0005
# EPOCHS = 20
# # Layer sizes for Deep Feature (DF) and Deep Interaction (DI) networks
# EMBEDDING_DIMS = [128, 64] 
# INTERACTION_DIMS = [64, 32]

# # --- 1. Data Loading ---
# def load_datasets():
#     print(f"Loading data from:\n Train: {TRAIN_PATH}\n Test:  {TEST_PATH}")
    
#     col_names = ['user_id', 'item_id', 'rating', 'timestamp']
#     train_df = pd.read_csv(TRAIN_PATH, sep='\t', names=col_names)
#     test_df = pd.read_csv(TEST_PATH, sep='\t', names=col_names)

#     # Combine to find total unique users/items for matrix dimensions
#     full_df = pd.concat([train_df, test_df])
    
#     # Create mappings to ensure IDs are 0-indexed and contiguous
#     user_ids = full_df['user_id'].unique()
#     item_ids = full_df['item_id'].unique()
    
#     user_to_idx = {u: i for i, u in enumerate(user_ids)}
#     item_to_idx = {m: i for i, m in enumerate(item_ids)}
    
#     # Apply mappings
#     train_df['user_idx'] = train_df['user_id'].map(user_to_idx)
#     train_df['item_idx'] = train_df['item_id'].map(item_to_idx)
#     test_df['user_idx'] = test_df['user_id'].map(user_to_idx)
#     test_df['item_idx'] = test_df['item_id'].map(item_to_idx)
    
#     # Adjust ratings to 0-4 scale for CrossEntropyLoss (class 0 = 1 star, class 4 = 5 stars)
#     train_df['rating'] = train_df['rating'] - 1
#     test_df['rating'] = test_df['rating'] - 1
    
#     num_users = len(user_ids)
#     num_items = len(item_ids)
    
#     return train_df, test_df, num_users, num_items

# # --- 2. Construct Rating Matrix (The Core J-NCF Input) ---
# def get_rating_matrix(df, num_users, num_items):
#     """
#     Creates the explicit rating matrix R based ONLY on training data.
#     R[u, i] = rating (1-5 scale) or 0 if unrated.
#     """
#     # Using float32 for CPU memory efficiency
#     R = torch.zeros((num_users, num_items), dtype=torch.float32)
    
#     # Iterate through training dataframe to fill matrix
#     # Note: We use +1 for rating to distinguish 1-star from 0 (missing)
#     users = torch.tensor(df['user_idx'].values)
#     items = torch.tensor(df['item_idx'].values)
#     ratings = torch.tensor(df['rating'].values) + 1 
    
#     R[users, items] = ratings.float()
    
#     return R

# # --- 3. PyTorch Dataset ---
# class MovieLensDataset(Dataset):
#     def __init__(self, df, rating_matrix):
#         self.users = torch.tensor(df['user_idx'].values, dtype=torch.long)
#         self.items = torch.tensor(df['item_idx'].values, dtype=torch.long)
#         self.labels = torch.tensor(df['rating'].values, dtype=torch.long)
#         self.R = rating_matrix

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         u = self.users[idx]
#         i = self.items[idx]
        
#         # J-NCF Inputs:
#         # v_u: The user's row from the rating matrix
#         # v_i: The item's column from the rating matrix
#         v_u = self.R[u]
#         v_i = self.R[:, i]
        
#         return v_u, v_i, self.labels[idx]

# # --- 4. J-NCF Model Architecture ---
# class JNCF_Net(nn.Module):
#     def __init__(self, num_users, num_items):
#         super(JNCF_Net, self).__init__()
        
#         # --- Deep Feature (DF) Network ---
#         # User tower (input size = number of items)
#         self.df_user = nn.Sequential(
#             nn.Linear(num_items, EMBEDDING_DIMS[0]),
#             nn.ReLU(),
#             nn.Linear(EMBEDDING_DIMS[0], EMBEDDING_DIMS[1]),
#             nn.ReLU()
#         )
        
#         # Item tower (input size = number of users)
#         self.df_item = nn.Sequential(
#             nn.Linear(num_users, EMBEDDING_DIMS[0]),
#             nn.ReLU(),
#             nn.Linear(EMBEDDING_DIMS[0], EMBEDDING_DIMS[1]),
#             nn.ReLU()
#         )
        
#         # --- Deep Interaction (DI) Network ---
#         # Input is concatenation of user and item features
#         di_input = EMBEDDING_DIMS[1] * 2
        
#         self.di_net = nn.Sequential(
#             nn.Linear(di_input, INTERACTION_DIMS[0]),
#             nn.ReLU(),
#             nn.Linear(INTERACTION_DIMS[0], INTERACTION_DIMS[1]),
#             nn.ReLU(),
#             # Output layer: 5 classes (for ratings 1, 2, 3, 4, 5)
#             nn.Linear(INTERACTION_DIMS[1], 5)
#         )

#     def forward(self, v_u, v_i):
#         # Extract features
#         z_u = self.df_user(v_u)
#         z_i = self.df_item(v_i)
        
#         # Concatenate features (Eq. 5 in paper: J-NCF_c)
#         combined = torch.cat((z_u, z_i), dim=1)
        
#         # Predict rating class logits
#         logits = self.di_net(combined)
#         return logits

# # --- 5. Main Execution ---
# def run():
#     print(f"Running on: {DEVICE}")
    
#     # 1. Load Data
#     train_df, test_df, num_users, num_items = load_datasets()
#     print(f"Users: {num_users}, Items: {num_items}")
#     print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

#     # 2. Build Rating Matrix (From Training Data ONLY)
#     # This matrix serves as the "Knowledge Base" for the DF network
#     print("Building Rating Matrix...")
#     R_train = get_rating_matrix(train_df, num_users, num_items).to(DEVICE)
    
#     # 3. Create DataLoaders
#     train_ds = MovieLensDataset(train_df, R_train)
#     test_ds = MovieLensDataset(test_df, R_train) # Note: Test set uses Train Matrix inputs to avoid leakage
    
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
#     # 4. Initialize Model
#     model = JNCF_Net(num_users, num_items).to(DEVICE)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
#     # 5. Training Loop
#     print("\nStarting Training...")
#     best_accuracy = 0.0
    
#     for epoch in range(EPOCHS):
#         model.train()
#         total_loss = 0
        
#         for batch_i, (v_u, v_i, labels) in enumerate(train_loader):
#             v_u, v_i, labels = v_u.to(DEVICE), v_i.to(DEVICE), labels.to(DEVICE)
            
#             optimizer.zero_grad()
#             outputs = model(v_u, v_i)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
            
#         avg_loss = total_loss / len(train_loader)
        
#         # 6. Evaluation (Calculate Accuracy)
#         model.eval()
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for v_u, v_i, labels in test_loader:
#                 v_u, v_i, labels = v_u.to(DEVICE), v_i.to(DEVICE), labels.to(DEVICE)
                
#                 outputs = model(v_u, v_i)
#                 _, predicted = torch.max(outputs.data, 1) # Get the class with highest probability
                
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
        
#         accuracy = correct / total
#         best_accuracy = max(best_accuracy, accuracy)
        
#         print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.4f}")

#     print("\n" + "="*30)
#     print(f"Final Best Accuracy: {best_accuracy:.4f}")
#     print("="*30)

# if __name__ == "__main__":
#     try:
#         run()
#     except FileNotFoundError:
#         print("\nError: Could not find files.")
#         print(f"Please check that '{TRAIN_PATH}' exists.")


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# --- Configuration ---
DEVICE = torch.device("cpu")

BASE_PATH = r"E:\NewDownloads\cf2\ml-100k"

BATCH_SIZE = 128
LEARNING_RATE = 0.0005
EPOCHS = 30

EMBEDDING_DIMS = [128, 64]
INTERACTION_DIMS = [64, 32]


# --- 1. Construct Rating Matrix ---
def get_rating_matrix(df, num_users, num_items):

    R = torch.zeros((num_users, num_items), dtype=torch.float32)

    users = torch.tensor(df['user_idx'].values)
    items = torch.tensor(df['item_idx'].values)
    ratings = torch.tensor(df['rating'].values) + 1

    R[users, items] = ratings.float()
    return R


# --- 2. PyTorch Dataset ---
class MovieLensDataset(Dataset):

    def __init__(self, df, rating_matrix):
        self.users = torch.tensor(df['user_idx'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_idx'].values, dtype=torch.long)
        self.labels = torch.tensor(df['rating'].values, dtype=torch.long)
        self.R = rating_matrix

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        u = self.users[idx]
        i = self.items[idx]

        v_u = self.R[u]
        v_i = self.R[:, i]

        return v_u, v_i, self.labels[idx]


# --- 3. J-NCF Model ---
class JNCF_Net(nn.Module):

    def __init__(self, num_users, num_items):
        super(JNCF_Net, self).__init__()

        self.df_user = nn.Sequential(
            nn.Linear(num_items, EMBEDDING_DIMS[0]),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIMS[0], EMBEDDING_DIMS[1]),
            nn.ReLU()
        )

        self.df_item = nn.Sequential(
            nn.Linear(num_users, EMBEDDING_DIMS[0]),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIMS[0], EMBEDDING_DIMS[1]),
            nn.ReLU()
        )

        di_input = EMBEDDING_DIMS[1] * 2

        self.di_net = nn.Sequential(
            nn.Linear(di_input, INTERACTION_DIMS[0]),
            nn.ReLU(),
            nn.Linear(INTERACTION_DIMS[0], INTERACTION_DIMS[1]),
            nn.ReLU(),
            nn.Linear(INTERACTION_DIMS[1], 5)
        )

    def forward(self, v_u, v_i):
        z_u = self.df_user(v_u)
        z_i = self.df_item(v_i)

        combined = torch.cat((z_u, z_i), dim=1)
        logits = self.di_net(combined)

        return logits


# --- 4. Main Execution (5-FOLD) ---
def run():

    print(f"Running on: {DEVICE}")

    folds = [1, 2, 3, 4, 5]
    fold_accuracies = []

    for fold in folds:

        print("\n" + "="*60)
        print(f"Starting Fold u{fold}")
        print("="*60)

        TRAIN_PATH = rf"{BASE_PATH}\u{fold}.base"
        TEST_PATH  = rf"{BASE_PATH}\u{fold}.test"

        col_names = ['user_id', 'item_id', 'rating', 'timestamp']

        train_df = pd.read_csv(TRAIN_PATH, sep='\t', names=col_names)
        test_df = pd.read_csv(TEST_PATH, sep='\t', names=col_names)

        full_df = pd.concat([train_df, test_df])

        user_ids = full_df['user_id'].unique()
        item_ids = full_df['item_id'].unique()

        user_to_idx = {u: i for i, u in enumerate(user_ids)}
        item_to_idx = {m: i for i, m in enumerate(item_ids)}

        train_df['user_idx'] = train_df['user_id'].map(user_to_idx)
        train_df['item_idx'] = train_df['item_id'].map(item_to_idx)

        test_df['user_idx'] = test_df['user_id'].map(user_to_idx)
        test_df['item_idx'] = test_df['item_id'].map(item_to_idx)

        train_df['rating'] = train_df['rating'] - 1
        test_df['rating'] = test_df['rating'] - 1

        num_users = len(user_ids)
        num_items = len(item_ids)

        print(f"Users: {num_users}, Items: {num_items}")
        print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

        # Build Rating Matrix
        print("Building Rating Matrix...")
        R_train = get_rating_matrix(train_df, num_users, num_items).to(DEVICE)

        # DataLoaders
        train_ds = MovieLensDataset(train_df, R_train)
        test_ds = MovieLensDataset(test_df, R_train)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Model
        model = JNCF_Net(num_users, num_items).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training
        print("\nStarting Training...")
        best_accuracy = 0.0

        for epoch in range(EPOCHS):

            model.train()
            total_loss = 0

            for v_u, v_i, labels in train_loader:

                v_u = v_u.to(DEVICE)
                v_i = v_i.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(v_u, v_i)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Evaluation
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for v_u, v_i, labels in test_loader:

                    v_u = v_u.to(DEVICE)
                    v_i = v_i.to(DEVICE)
                    labels = labels.to(DEVICE)

                    outputs = model(v_u, v_i)
                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            best_accuracy = max(best_accuracy, accuracy)

            print(
                f"Fold {fold} | Epoch {epoch+1:02d} | "
                f"Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.4f}"
            )

        print(f"\nBest Accuracy for Fold u{fold}: {best_accuracy:.4f}")
        fold_accuracies.append(best_accuracy)

    # ---- Final Report ----
    print("\n" + "="*60)
    print("5-Fold Accuracy Report")
    print("="*60)

    for i, acc in enumerate(fold_accuracies):
        print(f"Fold u{i+1} Accuracy: {acc:.4f}")

    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)

    print("\nOverall Average Accuracy:", round(avg_accuracy, 4))
    print("="*60)


if __name__ == "__main__":
    try:
        run()
    except FileNotFoundError:
        print("\nError: Could not find files.")
        print("Check BASE_PATH and fold filenames.")





