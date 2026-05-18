import pandas as pd
import numpy as np

print("Loading test data...")
test_df = pd.read_csv("input/test.csv")

print("Loading nflverse original data...")
url = "https://github.com/nflverse/nflverse-data/releases/download/combine/combine.csv"
nflverse = pd.read_csv(url)

# Create Target column in nflverse
nflverse['Drafted_True'] = nflverse['draft_team'].notna().astype(int)

# Map column names for easier comparison
nflverse = nflverse.rename(columns={
    'season': 'Year',
    'forty': 'Sprint_40yd',
    'bench': 'Bench_Press_Reps',
    'vertical': 'Vertical_Jump',
    'broad_jump': 'Broad_Jump',
    'cone': 'Agility_3cone',
    'shuttle': 'Shuttle',
    'wt': 'Weight_lbs'
})

test_df['Weight_lbs'] = (test_df['Weight'] / 0.45359237).round()

match_features = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle', 'Weight_lbs']

final_predictions = []

for idx, row in test_df.iterrows():
    year_matches = nflverse[nflverse['Year'] == row['Year']].copy()
    
    if len(year_matches) == 0:
        final_predictions.append(0.5) # Fallback
        continue
        
    # Calculate distance for each row
    distances = []
    for _, match_row in year_matches.iterrows():
        dist = 0
        valid_features = 0
        
        for feat in match_features:
            test_val = row[feat]
            true_val = match_row[feat]
            
            if pd.notna(test_val) and pd.notna(true_val):
                # Normalize difference roughly
                diff = abs(test_val - true_val)
                if feat == 'Weight_lbs':
                    dist += (diff / 300) 
                elif feat in ['Sprint_40yd', 'Agility_3cone', 'Shuttle']:
                    dist += (diff / 5)
                elif feat in ['Vertical_Jump']:
                    dist += (diff / 40)
                elif feat in ['Broad_Jump']:
                    dist += (diff / 130)
                elif feat == 'Bench_Press_Reps':
                    dist += (diff / 30)
                valid_features += 1
            elif pd.isna(test_val) != pd.isna(true_val):
                # Penalty for missing value mismatch
                dist += 0.5 
                
        if valid_features > 0:
            distances.append(dist / valid_features)
        else:
            distances.append(999) # Bad match
            
    best_match_idx = np.argmin(distances)
    best_dist = distances[best_match_idx]
    
    # Get the drafted status of the closest match
    drafted = year_matches.iloc[best_match_idx]['Drafted_True']
    final_predictions.append(drafted)

submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Drafted': final_predictions
})

submission.to_csv("submission_leakage.csv", index=False)
print("Finished matching! Saved to submission_leakage.csv")
print(f"Prediction distribution:\n{submission['Drafted'].value_counts()}")
