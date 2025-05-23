# %%


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# %%
# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%
class FootballDataset(Dataset):
    def __init__(self, features, home_goals, away_goals):
        self.features = torch.FloatTensor(features).to(device)
        self.home_goals = torch.FloatTensor(home_goals.values).reshape(-1, 1).to(device)
        self.away_goals = torch.FloatTensor(away_goals.values).reshape(-1, 1).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.home_goals[idx], self.away_goals[idx]

# %%

class EnhancedFootballPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.3):
        super(EnhancedFootballPredictor, self).__init__()
        
        # Build the shared layers with deeper architecture
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # Add batch normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Separate prediction heads with additional layers
        self.home_goals = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.BatchNorm1d(hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),  # Lower dropout in final layers
            nn.Linear(hidden_sizes[-1] // 2, 1)
        )
        
        self.away_goals = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.BatchNorm1d(hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_sizes[-1] // 2, 1)
        )
    
    def forward(self, x):
        # Process through shared layers
        shared_features = self.shared_layers(x)
        
        # Generate predictions
        home_goals = torch.relu(self.home_goals(shared_features))
        away_goals = torch.relu(self.away_goals(shared_features))
        
        return home_goals, away_goals



class EloRating:
    def __init__(self, base_rating=1500, k_factor=32, home_advantage=100):
        """
        Initialize the Elo rating system.
        
        Args:
            base_rating: Default starting rating for new teams
            k_factor: Determines how much ratings change after each match
            home_advantage: Rating points added to home team during calculation
        """
        self.base_rating = base_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings = {}
    
    def get_rating(self, team):
        """Get a team's current rating, or assign base rating if not present."""
        if team not in self.ratings:
            self.ratings[team] = self.base_rating
        return self.ratings[team]
    
    def calculate_expected_score(self, rating_a, rating_b):
        """Calculate the expected score (win probability) for team A."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    
    def update_ratings(self, home_team, away_team, result):
        """
        Update team ratings based on match result.
        
        Args:
            home_team: Name of the home team
            away_team: Name of the away team
            result: 1.0 for home win, 0.5 for draw, 0.0 for away win
        """
        # Get current ratings
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Calculate expected scores with home advantage
        home_expected = self.calculate_expected_score(
            home_rating + self.home_advantage, away_rating
        )
        away_expected = self.calculate_expected_score(
            away_rating, home_rating + self.home_advantage
        )
        
        # Update ratings
        self.ratings[home_team] += self.k_factor * (result - home_expected)
        self.ratings[away_team] += self.k_factor * ((1 - result) - away_expected)

def initialize_elo_ratings(df, team_to_id):
    """Initialize Elo ratings for all teams based on historical match data."""
    elo = EloRating(base_rating=1500, k_factor=32, home_advantage=100)
    
    # Process matches in chronological order
    for idx, match in df.sort_values('Date').iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Determine result
        if match['FTHG'] > match['FTAG']:
            result = 1.0  # Home win
        elif match['FTHG'] < match['FTAG']:
            result = 0.0  # Away win
        else:
            result = 0.5  # Draw
        
        # Update ratings
        elo.update_ratings(home_team, away_team, result)
    
    # Add Elo ratings to the team_to_id dictionary for reference
    for team in team_to_id:
        if team not in elo.ratings:
            elo.ratings[team] = elo.base_rating
    
    return elo

# %%
def create_enhanced_features(df, window=10, recent_weight=2.0):
    """
    Create enhanced features including Elo ratings and form metrics.
    
    Args:
        df: DataFrame containing match data
        window: Number of recent matches to consider for form calculations
        recent_weight: Weight multiplier for more recent matches
    """
    # Calculate basic statistics as before
    df['HomePoints'] = np.where(df['FTHG'] > df['FTAG'], 3, 
                              np.where(df['FTHG'] == df['FTAG'], 1, 0))
    df['AwayPoints'] = np.where(df['FTAG'] > df['FTHG'], 3, 
                              np.where(df['FTAG'] == df['FTHG'], 1, 0))
    
    # Initialize Elo rating system
    elo = EloRating(base_rating=1500, k_factor=32, home_advantage=100)
    
    # Add Elo ratings at match time (pre-match)
    df['HomeElo'] = 0.0
    df['AwayElo'] = 0.0
    df['EloDiff'] = 0.0  # Difference in Elo (Home - Away)
    
    # Process matches chronologically
    for idx, match in df.sort_values('Date').iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Store pre-match Elo ratings
        home_elo = elo.get_rating(home_team)
        away_elo = elo.get_rating(away_team)
        
        df.at[idx, 'HomeElo'] = home_elo
        df.at[idx, 'AwayElo'] = away_elo
        df.at[idx, 'EloDiff'] = home_elo - away_elo
        
        # Determine result and update ratings
        if match['FTHG'] > match['FTAG']:
            result = 1.0  # Home win
        elif match['FTHG'] < match['FTAG']:
            result = 0.0  # Away win
        else:
            result = 0.5  # Draw
        
        elo.update_ratings(home_team, away_team, result)
    
    # Add form features for each team
    for team in pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel('K')):
        # Get all matches where this team played
        team_data = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date')
        
        # Calculate points and goals
        team_data['TotalPoints'] = np.where(team_data['HomeTeam'] == team, 
                                          team_data['HomePoints'], 
                                          team_data['AwayPoints'])
        
        team_data['GoalsScored'] = np.where(team_data['HomeTeam'] == team, 
                                          team_data['FTHG'], 
                                          team_data['FTAG'])
        
        team_data['GoalsConceded'] = np.where(team_data['HomeTeam'] == team, 
                                            team_data['FTAG'], 
                                            team_data['FTHG'])
        
        # Calculate form metrics (weighted recent performance)
        team_data['RecentForm'] = team_data['TotalPoints'].rolling(window=window, min_periods=1).apply(
            lambda x: np.average(x, weights=np.linspace(1, recent_weight, len(x))))
        
        team_data['AttackStrength'] = team_data['GoalsScored'].rolling(window=window, min_periods=1).apply(
            lambda x: np.average(x, weights=np.linspace(1, recent_weight, len(x))))
        
        team_data['DefenseWeakness'] = team_data['GoalsConceded'].rolling(window=window, min_periods=1).apply(
            lambda x: np.average(x, weights=np.linspace(1, recent_weight, len(x))))
        
        # Standard cumulative features
        team_data['CumPoints'] = team_data['TotalPoints'].rolling(window=window, min_periods=1).sum()
        team_data['CumGoalsScored'] = team_data['GoalsScored'].rolling(window=window, min_periods=1).sum()
        team_data['CumGoalsConceded'] = team_data['GoalsConceded'].rolling(window=window, min_periods=1).sum()
        
        # Add win/loss streaks
        team_data['IsWin'] = (team_data['TotalPoints'] == 3).astype(int)
        team_data['IsLoss'] = (team_data['TotalPoints'] == 0).astype(int)
        team_data['WinStreak'] = team_data['IsWin'].rolling(window=5, min_periods=1).sum()
        team_data['LossStreak'] = team_data['IsLoss'].rolling(window=5, min_periods=1).sum()
        
        # Map these values back to the original dataframe
        for home_away in ['Home', 'Away']:
            condition = df[f'{home_away}Team'] == team
            df.loc[condition, f'{home_away}Team_RecentForm'] = team_data['RecentForm'].values
            df.loc[condition, f'{home_away}Team_AttackStrength'] = team_data['AttackStrength'].values
            df.loc[condition, f'{home_away}Team_DefenseWeakness'] = team_data['DefenseWeakness'].values
            df.loc[condition, f'{home_away}Team_WinStreak'] = team_data['WinStreak'].values
            df.loc[condition, f'{home_away}Team_LossStreak'] = team_data['LossStreak'].values
            df.loc[condition, f'{home_away}Team_CumPoints'] = team_data['CumPoints'].values
            df.loc[condition, f'{home_away}Team_CumGoalsScored'] = team_data['CumGoalsScored'].values
            df.loc[condition, f'{home_away}Team_CumGoalsConceded'] = team_data['CumGoalsConceded'].values
    
    # Add home advantage feature based on historical performance
    team_home_wins = df.groupby('HomeTeam')['HomePoints'].apply(lambda x: (x == 3).mean()).reset_index()
    team_home_wins.columns = ['Team', 'HomeWinRate']
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        home_win_rate = team_home_wins[team_home_wins['Team'] == home_team]['HomeWinRate'].values
        
        if len(home_win_rate) > 0:
            df.at[idx, 'HomeAdvantage'] = home_win_rate[0]
        else:
            df.at[idx, 'HomeAdvantage'] = 0.5  # Default value
    
    return df

# %%
class EnhancedFootballPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        super(EnhancedFootballPredictor, self).__init__()
        
        # Deeper shared network
        self.shared_layers = nn.Sequential(
            nn.BatchNorm1d(input_size),  # Normalize inputs
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),  # LeakyReLU to prevent dying neurons
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
        
        # Separate prediction heads with more layers
        self.home_goals = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size // 8, 1)
        )
        
        self.away_goals = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size // 8, 1)
        )
    
    def forward(self, x):
        # Process through shared layers
        shared_features = self.shared_layers(x)
        
        # Generate predictions
        home_goals = torch.relu(self.home_goals(shared_features))
        away_goals = torch.relu(self.away_goals(shared_features))
        
        return home_goals, away_goals


# %%
def calculate_expected_result(home_elo, away_elo, home_advantage=100):
    """Calculate expected result based on ELO ratings with home advantage."""
    # Add home advantage to home team's rating
    adjusted_home_elo = home_elo + home_advantage
    
    # Calculate expected outcome using the ELO formula
    exp_home = 1 / (1 + 10 ** ((away_elo - adjusted_home_elo) / 400))
    exp_away = 1 - exp_home
    
    return exp_home, exp_away

# %%
def train_enhanced_model(model, train_loader, val_loader, num_epochs=100, patience=10):
    """
    Train the model with enhanced training approach including learning rate scheduling
    and early stopping.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Maximum number of training epochs
        patience: Number of epochs to wait before early stopping
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    counter = 0  # Counter for early stopping
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_features, batch_home_goals, batch_away_goals in train_loader:
            optimizer.zero_grad()
            
            pred_home, pred_away = model(batch_features)
            loss_home = criterion(pred_home, batch_home_goals)
            loss_away = criterion(pred_away, batch_away_goals)
            
            # Combined loss with potential weighting
            loss = loss_home + loss_away
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        home_goals_mae = 0
        away_goals_mae = 0
        correct_outcomes = 0
        total_matches = 0
        
        with torch.no_grad():
            for batch_features, batch_home_goals, batch_away_goals in val_loader:
                pred_home, pred_away = model(batch_features)
                loss_home = criterion(pred_home, batch_home_goals)
                loss_away = criterion(pred_away, batch_away_goals)
                val_loss += (loss_home + loss_away).item()
                
                # Calculate MAE
                home_goals_mae += torch.mean(torch.abs(pred_home - batch_home_goals)).item()
                away_goals_mae += torch.mean(torch.abs(pred_away - batch_away_goals)).item()
                
                # Calculate match outcome accuracy
                total_matches += batch_home_goals.size(0)
                
                for i in range(batch_home_goals.size(0)):
                    actual_outcome = torch.sign(batch_home_goals[i] - batch_away_goals[i]).item()
                    predicted_outcome = torch.sign(pred_home[i] - pred_away[i]).item()
                    
                    if actual_outcome == predicted_outcome:
                        correct_outcomes += 1
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Calculate metrics
        home_goals_mae = home_goals_mae / len(val_loader)
        away_goals_mae = away_goals_mae / len(val_loader)
        outcome_accuracy = correct_outcomes / total_matches if total_matches > 0 else 0
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Home Goals MAE: {home_goals_mae:.2f}')
        print(f'Away Goals MAE: {away_goals_mae:.2f}')
        print(f'Outcome Accuracy: {outcome_accuracy:.2f}')
        print(f'Current LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pt')
            print("Model saved!")
        else:
            counter += 1
            print(f"No improvement for {counter} epochs")
        
        # Early stopping
        if counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return train_losses, val_losses, model


# %%
def calculate_actual_result(home_goals, away_goals):
    """Calculate actual result (1 for win, 0.5 for draw, 0 for loss)."""
    if home_goals > away_goals:
        return 1.0, 0.0  # Home win
    elif home_goals < away_goals:
        return 0.0, 1.0  # Away win
    else:
        return 0.5, 0.5  # Draw

# %%
def update_elo_ratings(home_team, away_team, home_goals, away_goals, elo_ratings, k_factor=20):
    """Update ELO ratings after a match."""
    home_elo = elo_ratings[home_team]
    away_elo = elo_ratings[away_team]
    
    # Calculate expected results
    exp_home, exp_away = calculate_expected_result(home_elo, away_elo)
    
    # Calculate actual results
    actual_home, actual_away = calculate_actual_result(home_goals, away_goals)
    
    # Calculate ELO changes
    home_elo_change = k_factor * (actual_home - exp_home)
    away_elo_change = k_factor * (actual_away - exp_away)
    
    # Update ELO ratings
    elo_ratings[home_team] = home_elo + home_elo_change
    elo_ratings[away_team] = away_elo + away_elo_change
    
    return elo_ratings

# %%
def enhanced_preprocess_data(df):
    """
    Enhanced preprocessing pipeline with feature selection.
    """
    teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel('K'))
    team_to_id = {team: idx for idx, team in enumerate(teams)}
    
    # Map team IDs
    df['HomeTeam_ID'] = df['HomeTeam'].map(team_to_id)
    df['AwayTeam_ID'] = df['AwayTeam'].map(team_to_id)
    
    # Feature list including Elo ratings and enhanced metrics
    features = [
        # Team IDs
        'HomeTeam_ID', 'AwayTeam_ID',
        
        # Elo ratings and difference
        'HomeElo', 'AwayElo', 'EloDiff',
        
        # Form metrics
        'HomeTeam_RecentForm', 'AwayTeam_RecentForm',
        'HomeTeam_WinStreak', 'AwayTeam_WinStreak',
        'HomeTeam_LossStreak', 'AwayTeam_LossStreak',
        
        # Attack and defense metrics
        'HomeTeam_AttackStrength', 'AwayTeam_AttackStrength',
        'HomeTeam_DefenseWeakness', 'AwayTeam_DefenseWeakness',
        
        # Cumulative statistics
        'HomeTeam_CumPoints', 'AwayTeam_CumPoints',
        'HomeTeam_CumGoalsScored', 'AwayTeam_CumGoalsScored',
        'HomeTeam_CumGoalsConceded', 'AwayTeam_CumGoalsConceded',
        
        # Home advantage
        'HomeAdvantage'
    ]
    
    # Ensure all features exist
    for feature in features:
        if feature not in df.columns:
            print(f"Warning: Feature {feature} not found in DataFrame")
            features.remove(feature)
    
    X = df[features].copy()
    y = df[['FTHG', 'FTAG']].copy()
    
    # Handle NaN values with median imputation
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    
    return X, y, team_to_id

# %%
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values('Date')
    return df


# %%
def predict_with_enhanced_model(model, scaler, team_to_id, elo_ratings, fixtures, df):
    """
    Make predictions using the enhanced model with Elo ratings.
    
    Args:
        model: Trained neural network model
        scaler: Feature scaler
        team_to_id: Dictionary mapping team names to IDs
        elo_ratings: EloRating instance with updated ratings
        fixtures: List of (home_team, away_team) tuples
        df: Original DataFrame with historical data
    """
    model.eval()
    predictions = []
    
    # Prepare home advantage dictionary
    team_home_wins = df.groupby('HomeTeam')['HomePoints'].apply(
        lambda x: (x == 3).mean()).reset_index()
    team_home_wins.columns = ['Team', 'HomeWinRate']
    team_home_advantage = dict(zip(team_home_wins['Team'], team_home_wins['HomeWinRate']))
    
    for home_team, away_team in fixtures:
        # Create feature vector for this fixture
        features = pd.DataFrame(columns=[
            'HomeTeam_ID', 'AwayTeam_ID', 'HomeElo', 'AwayElo', 'EloDiff',
            'HomeTeam_RecentForm', 'AwayTeam_RecentForm', 'HomeTeam_WinStreak', 'AwayTeam_WinStreak',
            'HomeTeam_LossStreak', 'AwayTeam_LossStreak', 'HomeTeam_AttackStrength', 
            'AwayTeam_AttackStrength', 'HomeTeam_DefenseWeakness', 'AwayTeam_DefenseWeakness',
            'HomeTeam_CumPoints', 'AwayTeam_CumPoints', 'HomeTeam_CumGoalsScored',
            'AwayTeam_CumGoalsScored', 'HomeTeam_CumGoalsConceded', 'AwayTeam_CumGoalsConceded',
            'HomeAdvantage'
        ])
        
        # Fill with default values
        features.loc[0] = 0
        
        # Add team IDs
        features.at[0, 'HomeTeam_ID'] = team_to_id.get(home_team, -1)
        features.at[0, 'AwayTeam_ID'] = team_to_id.get(away_team, -1)
        
        # Add Elo ratings
        home_elo = elo_ratings.get_rating(home_team)
        away_elo = elo_ratings.get_rating(away_team)
        features.at[0, 'HomeElo'] = home_elo
        features.at[0, 'AwayElo'] = away_elo
        features.at[0, 'EloDiff'] = home_elo - away_elo
        
        # Add home advantage
        features.at[0, 'HomeAdvantage'] = team_home_advantage.get(home_team, 0.5)
        
        # Add team metrics from the most recent games
        for prefix, team in [('Home', home_team), ('Away', away_team)]:
            team_data = df[df[f'{prefix}Team'] == team]
            if not team_data.empty:
                last_game = team_data.iloc[-1]
                
                # Feature mapping
                feature_mapping = {
                    f'{prefix}Team_RecentForm': f'{prefix}Team_RecentForm',
                    f'{prefix}Team_WinStreak': f'{prefix}Team_WinStreak',
                    f'{prefix}Team_LossStreak': f'{prefix}Team_LossStreak',
                    f'{prefix}Team_AttackStrength': f'{prefix}Team_AttackStrength',
                    f'{prefix}Team_DefenseWeakness': f'{prefix}Team_DefenseWeakness',
                    f'{prefix}Team_CumPoints': f'{prefix}Team_CumPoints',
                    f'{prefix}Team_CumGoalsScored': f'{prefix}Team_CumGoalsScored',
                    f'{prefix}Team_CumGoalsConceded': f'{prefix}Team_CumGoalsConceded'
                }
                
                # Copy values from last game
                for df_col, feat_col in feature_mapping.items():
                    if df_col in last_game:
                        features.at[0, feat_col] = last_game[df_col]
            else:
                print(f"Warning: No historical data found for {prefix.lower()} team {team}")
        
        # Scale features
        features_scaled = scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).to(device)
        
        # Make prediction
        with torch.no_grad():
            pred_home, pred_away = model(features_tensor)
            home_goals = round(float(pred_home.cpu().numpy().item()))
            away_goals = round(float(pred_away.cpu().numpy().item()))
            
            # Ensure non-negative
            home_goals = max(0, home_goals)
            away_goals = max(0, away_goals)
            
            # Determine result
            if home_goals > away_goals:
                result = f"{home_team} win"
            elif home_goals < away_goals:
                result = f"{away_team} win"
            else:
                result = "Draw"
            
            # Calculate win probabilities based on Elo
            home_win_prob = elo_ratings.calculate_expected_score(
                home_elo + elo_ratings.home_advantage, away_elo
            )
            draw_prob = 0.25  # Simplified estimate
            away_win_prob = 1 - home_win_prob - draw_prob
            
            predictions.append({
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'Predicted Score': f"{home_goals} - {away_goals}",
                'Predicted Result': result,
                'Home Goals': home_goals,
                'Away Goals': away_goals,
                'Home Win Prob': f"{home_win_prob:.2f}",
                'Draw Prob': f"{draw_prob:.2f}",
                'Away Win Prob': f"{away_win_prob:.2f}"
            })
    
    return predictions

# %%
def create_elo_features(df, initial_rating=1500):
    """Create ELO rating features based on historical match data."""
    # Get all unique teams
    teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel('K'))
    
    # Initialize ELO ratings
    elo_ratings = initialize_elo_ratings(teams, initial_rating)
    
    # Create columns for ELO ratings
    df['HomeTeam_ELO'] = 0.0
    df['AwayTeam_ELO'] = 0.0
    df['ELO_Difference'] = 0.0
    df['Expected_Home_Win'] = 0.0
    
    # Sort by date to process matches chronologically
    df = df.sort_values('Date')
    
    # Process matches one by one and update ELO ratings
    for idx, match in df.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Store current ELO ratings in the dataframe
        df.at[idx, 'HomeTeam_ELO'] = elo_ratings[home_team]
        df.at[idx, 'AwayTeam_ELO'] = elo_ratings[away_team]
        
        # Calculate ELO difference and expected win probability
        exp_home, _ = calculate_expected_result(elo_ratings[home_team], elo_ratings[away_team])
        df.at[idx, 'ELO_Difference'] = elo_ratings[home_team] - elo_ratings[away_team]
        df.at[idx, 'Expected_Home_Win'] = exp_home
        
        # Update ELO ratings after the match
        home_goals = match['FTHG']
        away_goals = match['FTAG']
        elo_ratings = update_elo_ratings(home_team, away_team, home_goals, away_goals, elo_ratings)
    
    return df

# %%

def create_historical_features(df, window=10):
    df['HomePoints'] = np.where(df['FTHG'] > df['FTAG'], 3, 
                               np.where(df['FTHG'] == df['FTAG'], 1, 0))
    df['AwayPoints'] = np.where(df['FTAG'] > df['FTHG'], 3, 
                               np.where(df['FTAG'] == df['FTHG'], 1, 0))
    
    for team in pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel('K')):
        team_data = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date')
        
        team_data['TotalPoints'] = np.where(team_data['HomeTeam'] == team, 
                                          team_data['HomePoints'], 
                                          team_data['AwayPoints'])
        team_data['GoalsScored'] = np.where(team_data['HomeTeam'] == team, 
                                          team_data['FTHG'], 
                                          team_data['FTAG'])
        team_data['GoalsConceded'] = np.where(team_data['HomeTeam'] == team, 
                                            team_data['FTAG'], 
                                            team_data['FTHG'])
        
        team_data['CumPoints'] = team_data['TotalPoints'].rolling(window=window, min_periods=1).sum()
        team_data['CumGoalsScored'] = team_data['GoalsScored'].rolling(window=window, min_periods=1).sum()
        team_data['CumGoalsConceded'] = team_data['GoalsConceded'].rolling(window=window, min_periods=1).sum()
        
        df.loc[(df['HomeTeam'] == team), 'HomeTeam_CumPoints'] = team_data['CumPoints']
        df.loc[(df['AwayTeam'] == team), 'AwayTeam_CumPoints'] = team_data['CumPoints']
        df.loc[(df['HomeTeam'] == team), 'HomeTeam_CumGoalsScored'] = team_data['CumGoalsScored']
        df.loc[(df['AwayTeam'] == team), 'AwayTeam_CumGoalsScored'] = team_data['CumGoalsScored']
        df.loc[(df['HomeTeam'] == team), 'HomeTeam_CumGoalsConceded'] = team_data['CumGoalsConceded']
        df.loc[(df['AwayTeam'] == team), 'AwayTeam_CumGoalsConceded'] = team_data['CumGoalsConceded']
    
    return df

# %%
def add_form_features(df, form_window=5):
    """Add team form features based on recent results."""
    # Get all unique teams
    teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel('K'))
    
    # Initialize form features
    df['HomeTeam_Form'] = 0.0
    df['AwayTeam_Form'] = 0.0
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Process each team
    for team in teams:
        # Get all matches for this team
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
        
        # Calculate result from team's perspective (1 for win, 0.5 for draw, 0 for loss)
        team_matches['TeamResult'] = 0.0
        
        # Home matches
        home_idx = team_matches[team_matches['HomeTeam'] == team].index
        team_matches.loc[home_idx, 'TeamResult'] = np.where(
            team_matches.loc[home_idx, 'FTHG'] > team_matches.loc[home_idx, 'FTAG'], 1.0,
            np.where(team_matches.loc[home_idx, 'FTHG'] == team_matches.loc[home_idx, 'FTAG'], 0.5, 0.0)
        )
        
        # Away matches
        away_idx = team_matches[team_matches['AwayTeam'] == team].index
        team_matches.loc[away_idx, 'TeamResult'] = np.where(
            team_matches.loc[away_idx, 'FTAG'] > team_matches.loc[away_idx, 'FTHG'], 1.0,
            np.where(team_matches.loc[away_idx, 'FTAG'] == team_matches.loc[away_idx, 'FTHG'], 0.5, 0.0)
        )
        
        # Calculate rolling form
        team_matches['Form'] = team_matches['TeamResult'].rolling(window=form_window, min_periods=1).mean()
        
        # Update form in original dataframe
        for idx, match in team_matches.iterrows():
            if match['HomeTeam'] == team:
                df.at[idx, 'HomeTeam_Form'] = match['Form']
            else:
                df.at[idx, 'AwayTeam_Form'] = match['Form']
    
    return df

def add_rolling_averages(df, window=5):
    """Add rolling averages for goals scored and conceded."""
    teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel('K'))
    
    # Initialize new columns
    df['HomeTeam_AvgGoalsScored'] = 0.0
    df['HomeTeam_AvgGoalsConceded'] = 0.0
    df['AwayTeam_AvgGoalsScored'] = 0.0
    df['AwayTeam_AvgGoalsConceded'] = 0.0
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Process each team
    for team in teams:
        # Get all matches for this team
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
        
        # Add goals scored and conceded from team perspective
        team_matches['TeamGoalsScored'] = 0
        team_matches['TeamGoalsConceded'] = 0
        
        # Home matches
        home_idx = team_matches[team_matches['HomeTeam'] == team].index
        team_matches.loc[home_idx, 'TeamGoalsScored'] = team_matches.loc[home_idx, 'FTHG']
        team_matches.loc[home_idx, 'TeamGoalsConceded'] = team_matches.loc[home_idx, 'FTAG']
        
        # Away matches
        away_idx = team_matches[team_matches['AwayTeam'] == team].index
        team_matches.loc[away_idx, 'TeamGoalsScored'] = team_matches.loc[away_idx, 'FTAG']
        team_matches.loc[away_idx, 'TeamGoalsConceded'] = team_matches.loc[away_idx, 'FTHG']
        
        # Calculate rolling averages
        team_matches['AvgGoalsScored'] = team_matches['TeamGoalsScored'].rolling(window=window, min_periods=1).mean()
        team_matches['AvgGoalsConceded'] = team_matches['TeamGoalsConceded'].rolling(window=window, min_periods=1).mean()
        
        # Update in original dataframe
        for idx, match in team_matches.iterrows():
            if match['HomeTeam'] == team:
                df.at[idx, 'HomeTeam_AvgGoalsScored'] = match['AvgGoalsScored']
                df.at[idx, 'HomeTeam_AvgGoalsConceded'] = match['AvgGoalsConceded']
            else:
                df.at[idx, 'AwayTeam_AvgGoalsScored'] = match['AvgGoalsScored']
                df.at[idx, 'AwayTeam_AvgGoalsConceded'] = match['AvgGoalsConceded']
    
    return df

# %%
def create_enhanced_features(df, window=10):
    """Create enhanced features for improved predictions."""
    # Call the original historical features function
    df = create_historical_features(df, window)
    
    # Add ELO ratings
    df = create_elo_features(df)
    
    # Add recent form (last 5 games)
    df = add_form_features(df, form_window=5)
    
    # Add goal difference features
    df['HomeTeam_GoalDiff'] = df['HomeTeam_CumGoalsScored'] - df['HomeTeam_CumGoalsConceded']
    df['AwayTeam_GoalDiff'] = df['AwayTeam_CumGoalsScored'] - df['AwayTeam_CumGoalsConceded']
    
    # Add rolling averages for goals scored and conceded
    df = add_rolling_averages(df, window=5)
    
    # Add head-to-head statistics
    df = add_head_to_head_stats(df)
    
    # Add seasonal context (e.g., how far into the season)
    df['MatchNumber'] = df.groupby(['HomeTeam', 'Season']).cumcount() + 1
    df['SeasonProgress'] = df['MatchNumber'] / df.groupby('Season')['MatchNumber'].transform('max')
    
    return df

# %%
def add_head_to_head_stats(df):
    """Add head-to-head statistics between teams."""
    # Initialize H2H columns
    df['H2H_HomeWins'] = 0
    df['H2H_AwayWins'] = 0
    df['H2H_Draws'] = 0
    df['H2H_TotalMatches'] = 0
    df['H2H_HomeGoalsAvg'] = 0.0
    df['H2H_AwayGoalsAvg'] = 0.0
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Process each match
    for idx, current_match in df.iterrows():
        home_team = current_match['HomeTeam']
        away_team = current_match['AwayTeam']
        match_date = current_match['Date']
        
        # Find all previous matches between these teams
        h2h_matches = df[
            ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team) |
             (df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team)) &
            (df['Date'] < match_date)
        ]
        
        if len(h2h_matches) > 0:
            # Direct H2H with current home team at home
            direct_h2h = h2h_matches[(h2h_matches['HomeTeam'] == home_team) & 
                                     (h2h_matches['AwayTeam'] == away_team)]
            
            # Reverse H2H with current away team at home
            reverse_h2h = h2h_matches[(h2h_matches['HomeTeam'] == away_team) & 
                                      (h2h_matches['AwayTeam'] == home_team)]
            
            # Calculate statistics
            home_wins = len(direct_h2h[direct_h2h['FTHG'] > direct_h2h['FTAG']]) + \
                        len(reverse_h2h[reverse_h2h['FTAG'] > reverse_h2h['FTHG']])
            
            away_wins = len(direct_h2h[direct_h2h['FTHG'] < direct_h2h['FTAG']]) + \
                        len(reverse_h2h[reverse_h2h['FTAG'] < reverse_h2h['FTHG']])
            
            draws = len(direct_h2h[direct_h2h['FTHG'] == direct_h2h['FTAG']]) + \
                    len(reverse_h2h[reverse_h2h['FTAG'] == reverse_h2h['FTHG']])
            
            total_matches = len(h2h_matches)
            
            # Calculate average goals
            home_goals_avg = (direct_h2h['FTHG'].mean() if len(direct_h2h) > 0 else 0) + \
                             (reverse_h2h['FTAG'].mean() if len(reverse_h2h) > 0 else 0)
            
            away_goals_avg = (direct_h2h['FTAG'].mean() if len(direct_h2h) > 0 else 0) + \
                             (reverse_h2h['FTHG'].mean() if len(reverse_h2h) > 0 else 0)
            
            if len(direct_h2h) > 0 and len(reverse_h2h) > 0:
                home_goals_avg /= 2
                away_goals_avg /= 2
            
            # Update the dataframe
            df.at[idx, 'H2H_HomeWins'] = home_wins
            df.at[idx, 'H2H_AwayWins'] = away_wins
            df.at[idx, 'H2H_Draws'] = draws
            df.at[idx, 'H2H_TotalMatches'] = total_matches
            df.at[idx, 'H2H_HomeGoalsAvg'] = home_goals_avg
            df.at[idx, 'H2H_AwayGoalsAvg'] = away_goals_avg
    
    return df

# %%


# %%
def preprocess_data(df):
    teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel('K'))
    team_to_id = {team: idx for idx, team in enumerate(teams)}
    
    df['HomeTeam_ID'] = df['HomeTeam'].map(team_to_id)
    df['AwayTeam_ID'] = df['AwayTeam'].map(team_to_id)
    
    features = ['HomeTeam_ID', 'AwayTeam_ID', 'HomeTeam_CumPoints', 'AwayTeam_CumPoints',
                'HomeTeam_CumGoalsScored', 'AwayTeam_CumGoalsScored',
                'HomeTeam_CumGoalsConceded', 'AwayTeam_CumGoalsConceded']
    
    X = df[features]
    y = df[['FTHG', 'FTAG']]
    
    return X, y, team_to_id

# %%
def train_model_with_early_stopping(model, train_loader, val_loader, patience=10, num_epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added L2 regularization
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    no_improve_count = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_features, batch_home_goals, batch_away_goals in train_loader:
            optimizer.zero_grad()
            
            pred_home, pred_away = model(batch_features)
            loss_home = criterion(pred_home, batch_home_goals)
            loss_away = criterion(pred_away, batch_away_goals)
            
            # Optional: Add custom loss component to encourage realistic scores
            pred_home_rounded = torch.round(pred_home)
            pred_away_rounded = torch.round(pred_away)
            realistic_loss = torch.mean(torch.abs(pred_home_rounded - pred_home) + 
                                       torch.abs(pred_away_rounded - pred_away))
            
            loss = loss_home + loss_away + 0.1 * realistic_loss
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        home_goals_mae = 0
        away_goals_mae = 0
        
        with torch.no_grad():
            for batch_features, batch_home_goals, batch_away_goals in val_loader:
                pred_home, pred_away = model(batch_features)
                loss_home = criterion(pred_home, batch_home_goals)
                loss_away = criterion(pred_away, batch_away_goals)
                val_loss += (loss_home + loss_away).item()
                
                home_goals_mae += torch.mean(torch.abs(pred_home - batch_home_goals)).item()
                away_goals_mae += torch.mean(torch.abs(pred_away - batch_away_goals)).item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate adjustment
        scheduler.step(val_loss)
        
        # Calculate MAE metrics
        home_goals_mae = home_goals_mae / len(val_loader)
        away_goals_mae = away_goals_mae / len(val_loader)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Home Goals MAE: {home_goals_mae:.2f}')
        print(f'Away Goals MAE: {away_goals_mae:.2f}')
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"New best model saved!")
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} epochs")
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model
    model.load_state_dict(best_model_state)
    return train_losses, val_losses, model

# %%
def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# %%
def predict_weekend_fixtures(model, scaler, team_to_id, fixtures, df):
    def create_match_features(home_team, away_team, team_to_id, df):
        features = pd.DataFrame(columns=['HomeTeam_ID', 'AwayTeam_ID', 'HomeTeam_CumPoints',
                                       'AwayTeam_CumPoints', 'HomeTeam_CumGoalsScored',
                                       'AwayTeam_CumGoalsScored', 'HomeTeam_CumGoalsConceded',
                                       'AwayTeam_CumGoalsConceded'])
        
        # Initialize with default values
        default_values = {
            'HomeTeam_CumPoints': 0.0,
            'AwayTeam_CumPoints': 0.0,
            'HomeTeam_CumGoalsScored': 0.0,
            'AwayTeam_CumGoalsScored': 0.0,
            'HomeTeam_CumGoalsConceded': 0.0,
            'AwayTeam_CumGoalsConceded': 0.0
        }
        
        features.loc[0] = default_values
        features.at[0, 'HomeTeam_ID'] = team_to_id.get(home_team, -1)
        features.at[0, 'AwayTeam_ID'] = team_to_id.get(away_team, -1)
        
        # Get home team stats
        home_data = df[df['HomeTeam'] == home_team]
        if not home_data.empty:
            last_home_game = home_data.iloc[-1]
            features.at[0, 'HomeTeam_CumPoints'] = last_home_game['HomeTeam_CumPoints']
            features.at[0, 'HomeTeam_CumGoalsScored'] = last_home_game['HomeTeam_CumGoalsScored']
            features.at[0, 'HomeTeam_CumGoalsConceded'] = last_home_game['HomeTeam_CumGoalsConceded']
        else:
            print(f"Warning: No historical data found for home team {home_team}")
            
        # Get away team stats    
        away_data = df[df['AwayTeam'] == away_team]
        if not away_data.empty:
            last_away_game = away_data.iloc[-1]
            features.at[0, 'AwayTeam_CumPoints'] = last_away_game['AwayTeam_CumPoints']
            features.at[0, 'AwayTeam_CumGoalsScored'] = last_away_game['AwayTeam_CumGoalsScored']
            features.at[0, 'AwayTeam_CumGoalsConceded'] = last_away_game['AwayTeam_CumGoalsConceded']
        else:
            print(f"Warning: No historical data found for away team {away_team}")
        
        return features

    model.eval()
    predictions = []
    
    for home_team, away_team in fixtures:
        features = create_match_features(home_team, away_team, team_to_id, df)
        features_scaled = scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).to(device)
        
        with torch.no_grad():
            pred_home, pred_away = model(features_tensor)
            # Fixed the deprecation warnings by properly extracting single values
            home_goals = round(float(pred_home.cpu().numpy().item()))
            away_goals = round(float(pred_away.cpu().numpy().item()))
            
            if home_goals > away_goals:
                result = f"{home_team} win"
            elif home_goals < away_goals:
                result = f"{away_team} win"
            else:
                result = "Draw"
            
            predictions.append({
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'Predicted Score': f"{home_goals} - {away_goals}",
                'Predicted Result': result,
                'Home Goals': home_goals,
                'Away Goals': away_goals
            })
    
    return predictions

# %%
def generate_league_table(model, scaler, team_to_id, df):
    # Get all unique teams
    teams = list(team_to_id.keys())
    team_stats = {team: {
        'Played': 0,
        'Won': 0,
        'Drawn': 0,
        'Lost': 0,
        'GF': 0,
        'GA': 0,
        'GD': 0,
        'Points': 0
    } for team in teams}
    
    # Generate all possible fixtures (home and away)
    for home_team in teams:
        for away_team in teams:
            if home_team != away_team:
                # Prepare match features
                features = prepare_match_features(home_team, away_team, team_to_id, df)
                features_scaled = scaler.transform(features)
                features_tensor = torch.FloatTensor(features_scaled).to(device)
                
                # Make prediction
                model.eval()
                with torch.no_grad():
                    pred_home, pred_away = model(features_tensor)
                    home_goals = round(float(pred_home.cpu().numpy().item()))
                    away_goals = round(float(pred_away.cpu().numpy().item()))
                
                # Update stats for both teams
                team_stats[home_team]['Played'] += 1
                team_stats[away_team]['Played'] += 1
                
                team_stats[home_team]['GF'] += home_goals
                team_stats[home_team]['GA'] += away_goals
                team_stats[away_team]['GF'] += away_goals
                team_stats[away_team]['GA'] += home_goals
                
                # Update points and results
                if home_goals > away_goals:
                    team_stats[home_team]['Won'] += 1
                    team_stats[home_team]['Points'] += 3
                    team_stats[away_team]['Lost'] += 1
                elif away_goals > home_goals:
                    team_stats[away_team]['Won'] += 1
                    team_stats[away_team]['Points'] += 3
                    team_stats[home_team]['Lost'] += 1
                else:
                    team_stats[home_team]['Drawn'] += 1
                    team_stats[away_team]['Drawn'] += 1
                    team_stats[home_team]['Points'] += 1
                    team_stats[away_team]['Points'] += 1
    
    # Calculate goal differences
    for team in team_stats:
        team_stats[team]['GD'] = team_stats[team]['GF'] - team_stats[team]['GA']
    
    # Convert to DataFrame and sort
    table = pd.DataFrame.from_dict(team_stats, orient='index')
    table['Team'] = table.index
    table = table[['Team', 'Played', 'Won', 'Drawn', 'Lost', 'GF', 'GA', 'GD', 'Points']]
    table = table.sort_values(['Points', 'GD', 'GF'], ascending=[False, False, False]).reset_index(drop=True)
    
    # Save table to CSV
    table.to_csv('predicted_league_table.csv', index=False)
    
    # Display table with formatting
    pd.set_option('display.max_rows', None)
    print("\nPredicted Final League Table:")
    print("=" * 80)
    print(f"{'Pos':>3} {'Team':<25} {'P':>3} {'W':>3} {'D':>3} {'L':>3} {'GF':>4} {'GA':>4} {'GD':>4} {'Pts':>4}")
    print("-" * 80)
    
    for idx, row in table.iterrows():
        pos = idx + 1
        print(f"{pos:3d} {row['Team']:<25} {row['Played']:3d} {row['Won']:3d} {row['Drawn']:3d} "
              f"{row['Lost']:3d} {row['GF']:4d} {row['GA']:4d} {row['GD']:4d} {row['Points']:4d}")
    
    print("=" * 80)
    
    # Print additional statistics
    print("\nLeague Statistics:")
    print(f"Total Goals: {table['GF'].sum()}")
    print(f"Average Goals per Game: {table['GF'].sum() / (table['Played'].sum() / 2):.2f}")
    print(f"Average Points: {table['Points'].mean():.1f}")
    
    # Identify promotion and relegation zones
    print("\nPromotion Zone:")
    print(table.head(3))
    print("\nRelegation Zone:")
    print(table.tail(3))
    
    return table


# %%
# Add this helper function to prepare match features
def prepare_match_features(home_team, away_team, team_to_id, df):
    features = pd.DataFrame(columns=['HomeTeam_ID', 'AwayTeam_ID', 'HomeTeam_CumPoints',
                                   'AwayTeam_CumPoints', 'HomeTeam_CumGoalsScored',
                                   'AwayTeam_CumGoalsScored', 'HomeTeam_CumGoalsConceded',
                                   'AwayTeam_CumGoalsConceded'])
    
    features.loc[0] = 0  # Initialize with zeros
    features.at[0, 'HomeTeam_ID'] = team_to_id.get(home_team, -1)
    features.at[0, 'AwayTeam_ID'] = team_to_id.get(away_team, -1)
    
    # Get team stats from their most recent games
    home_data = df[df['HomeTeam'] == home_team]
    away_data = df[df['AwayTeam'] == away_team]
    
    if not home_data.empty:
        last_home_game = home_data.iloc[-1]
        features.at[0, 'HomeTeam_CumPoints'] = last_home_game['HomeTeam_CumPoints']
        features.at[0, 'HomeTeam_CumGoalsScored'] = last_home_game['HomeTeam_CumGoalsScored']
        features.at[0, 'HomeTeam_CumGoalsConceded'] = last_home_game['HomeTeam_CumGoalsConceded']
    
    if not away_data.empty:
        last_away_game = away_data.iloc[-1]
        features.at[0, 'AwayTeam_CumPoints'] = last_away_game['AwayTeam_CumPoints']
        features.at[0, 'AwayTeam_CumGoalsScored'] = last_away_game['AwayTeam_CumGoalsScored']
        features.at[0, 'AwayTeam_CumGoalsConceded'] = last_away_game['AwayTeam_CumGoalsConceded']
    
    return features

# %%
def save_predictions(predictions, filename='weekend_predictions.csv'):
    # Save to CSV
    df = pd.DataFrame(predictions)
    df.to_csv(filename, index=False)
    
    # Print formatted predictions table
    print("\nWeekend Predictions:")
    print("-" * 70)
    
    # Create formatted header
    print(f"{'Home Team':<20} {'Away Team':<20} {'Score':<10} {'Prediction':<20}")
    print("-" * 70)
    
    # Print each prediction in formatted rows
    for pred in predictions:
        home_team = pred['HomeTeam'][:20]  # Truncate if too long
        away_team = pred['AwayTeam'][:20]  # Truncate if too long
        score = pred['Predicted Score']
        result = pred['Predicted Result']
        
        print(f"{home_team:<20} {away_team:<20} {score:<10} {result:<20}")
    
    print("-" * 70)
    
    # Print statistics
    total_games = len(predictions)
    home_wins = sum(1 for p in predictions if p['HomeTeam'] in p['Predicted Result'])
    away_wins = sum(1 for p in predictions if p['AwayTeam'] in p['Predicted Result'])
    draws = sum(1 for p in predictions if 'Draw' in p['Predicted Result'])
    
    print(f"\nStatistics:")
    print(f"Home Wins: {home_wins} ({home_wins/total_games*100:.1f}%)")
    print(f"Away Wins: {away_wins} ({away_wins/total_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/total_games*100:.1f}%)")

# %%

# %%
from sklearn.model_selection import KFold

def cross_validate_model(X, y, num_folds=5, batch_size=32, epochs=50, patience=10):
    """Perform k-fold cross-validation."""
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_predictions = []
    all_actuals = []
    
    # Scale once using all data to ensure consistent scaling across folds
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nTraining Fold {fold+1}/{num_folds}")
        
        # Split data
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create datasets and dataloaders
        train_dataset = FootballDataset(X_train, y_train['FTHG'], y_train['FTAG'])
        val_dataset = FootballDataset(X_val, y_val['FTHG'], y_val['FTAG'])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        input_size = X.shape[1]
        model = EnhancedFootballPredictor(input_size).to(device)
        
        # Train model with early stopping
        _, _, trained_model = train_model_with_early_stopping(
            model, train_loader, val_loader, patience=patience, num_epochs=epochs
        )
        
        # Evaluate on validation set
        trained_model.eval()
        val_predictions = []
        
        with torch.no_grad():
            for batch_features, batch_home_goals, batch_away_goals in val_loader:
                pred_home, pred_away = trained_model(batch_features)
                
                for i in range(len(pred_home)):
                    pred = {
                        'home_goals_pred': float(pred_home[i].cpu().numpy()),
                        'away_goals_pred': float(pred_away[i].cpu().numpy()),
                        'home_goals_actual': float(batch_home_goals[i].cpu().numpy()),
                        'away_goals_actual': float(batch_away_goals[i].cpu().numpy())
                    }
                    val_predictions.append(pred)
        
        # Calculate metrics
        home_mae = np.mean([abs(p['home_goals_pred'] - p['home_goals_actual']) for p in val_predictions])
        away_mae = np.mean([abs(p['away_goals_pred'] - p['away_goals_actual']) for p in val_predictions])
        
        # Calculate accuracy of win/draw/loss prediction
        correct_predictions = 0
        for p in val_predictions:
            pred_result = np.sign(round(p['home_goals_pred']) - round(p['away_goals_pred']))
            actual_result = np.sign(p['home_goals_actual'] - p['away_goals_actual'])
            
            if pred_result == actual_result:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(val_predictions)
        
        fold_results.append({
            'fold': fold + 1,
            'home_mae': home_mae,
            'away_mae': away_mae,
            'accuracy': accuracy
        })
        
        all_predictions.extend(val_predictions)
        
        print(f"Fold {fold+1} Results:")
        print(f"Home Goals MAE: {home_mae:.3f}")
        print(f"Away Goals MAE: {away_mae:.3f}")
        print(f"Match Outcome Accuracy: {accuracy:.3f}")
    
    # Calculate overall metrics
    overall_home_mae = np.mean([r['home_mae'] for r in fold_results])
    overall_away_mae = np.mean([r['away_mae'] for r in fold_results])
    overall_accuracy = np.mean([r['accuracy'] for r in fold_results])
    
    print("\nCross-Validation Results:")
    print(f"Average Home Goals MAE: {overall_home_mae:.3f}")
    print(f"Average Away Goals MAE: {overall_away_mae:.3f}")
    print(f"Average Match Outcome Accuracy: {overall_accuracy:.3f}")
    
    return fold_results, scaler

# %%
def train_final_model(X, y, scaler, batch_size=32, epochs=100, patience=15):
    """Train the final model on all data after cross-validation."""
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Create dataset and dataloader for all data
    dataset = FootballDataset(X_scaled, y['FTHG'], y['FTAG'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_size = X.shape[1]
    model = EnhancedFootballPredictor(input_size).to(device)
    
    # Train model on all data
    print("\nTraining final model on all data...")
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_features, batch_home_goals, batch_away_goals in dataloader:
            optimizer.zero_grad()
            
            pred_home, pred_away = model(batch_features)
            loss_home = criterion(pred_home, batch_home_goals)
            loss_away = criterion(pred_away, batch_away_goals)
            loss = loss_home + loss_away
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save the final model
    torch.save(model.state_dict(), 'final_model.pt')
    print("Final model")

# %%
def main():
    # Load and prepare data
    filepath = "E1(2).csv"  # Replace with your data file path
    df = load_data(filepath)
    df = create_historical_features(df)
    X, y, team_to_id = preprocess_data(df)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create datasets and dataloaders
    train_dataset = FootballDataset(X_train_scaled, y_train['FTHG'], y_train['FTAG'])
    val_dataset = FootballDataset(X_val_scaled, y_val['FTHG'], y_val['FTAG'])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize and train model
    model = FootballPredictor(input_size=X_train.shape[1]).to(device)
    train_losses, val_losses = train_model(model, train_loader, val_loader)
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    # Load best model for predictions
    model.load_state_dict(torch.load('best_model.pt'))
    
    weekend_fixtures = [
    ('Blackburn Rovers', 'Sheffield United'),
    ('Oxford United', 'Swansea City'),
    ('Stoke City', 'Derby County'),
    ('Cardiff City', 'Norwich City'),
    ('Hull City', 'Portsmouth FC'),
    ('Leeds United', 'Plymouth Argyle'),
    ('Middlesbrough FC', 'Coventry City'),
    ('Preston North End', 'Bristol City'),
    ('Queens Park Rangers', 'Sunderland AFC'),
    ('Sheffield Wednesday', 'Watford FC')
    ]
    # Make predictions for weekend fixtures
    predictions = predict_weekend_fixtures(model, scaler, team_to_id, weekend_fixtures, df)
    save_predictions(predictions)
    
    print("\nWeekend Predictions:")
    for pred in predictions:
        print(f"{pred['HomeTeam']} vs {pred['AwayTeam']}: {pred['Predicted Score']} ({pred['Predicted Result']})")
    
    return model, scaler, team_to_id, df

# %%
if __name__ == "__main__":
    model, scaler, team_to_id, df = main()

# %%
def main():
    # Load and prepare data
    filepath = "E1(2).csv"  # Replace with your data file path
    df = load_data(filepath)
    
    # Add Elo ratings and enhanced features
    df = create_enhanced_features(df, window=10) 
    # Preprocess data
    X, y, team_to_id = enhanced_preprocess_data(df)
    
    # Initialize Elo ratings
    elo_ratings = initialize_elo_ratings(df, team_to_id)
    
    # Split data chronologically (important for time series data)
    # Use the most recent 20% of matches for validation
    train_size = int(0.8 * len(df))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create datasets and dataloaders
    train_dataset = FootballDataset(X_train_scaled, y_train['FTHG'], y_train['FTAG'])
    val_dataset = FootballDataset(X_val_scaled, y_val['FTHG'], y_val['FTAG'])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize enhanced model
    model = EnhancedFootballPredictor(input_size=X_train.shape[1]).to(device)
    
    # Train model with enhanced training function
    train_losses, val_losses, model = train_enhanced_model(
        model, train_loader, val_loader, num_epochs=100, patience=15
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    # Example weekend fixtures
    weekend_fixtures = [
        ('Blackburn Rovers', 'Sheffield United'),
        ('Oxford United', 'Swansea City'),
        ('Stoke City', 'Derby County'),
        ('Cardiff City', 'Norwich City'),
        ('Hull City', 'Portsmouth FC'),
        ('Leeds United', 'Plymouth Argyle'),
        ('Middlesbrough FC', 'Coventry City'),
        ('Preston North End', 'Bristol City'),
        ('Queens Park Rangers', 'Sunderland AFC'),
        ('Sheffield Wednesday', 'Watford FC')
    ]
    
    # Make predictions with enhanced model
    predictions = predict_with_enhanced_model(
        model, scaler, team_to_id, elo_ratings, weekend_fixtures, df
    )
    
    # Save and display predictions
    save_predictions(predictions, filename='enhanced_predictions.csv')
    


    # Save model for future use
    torch.save(model.state_dict(), 'enhanced_football_model.pth')
    print("Model saved successfully!")



# %%
if __name__ == "__main__":
    main()


