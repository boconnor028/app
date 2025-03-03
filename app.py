import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import joblib
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

# API Key
API_KEY = "49b3a4534bff42d590f2ceba0ae69162"

# Set season and date dynamically
SEASON = "2025"
GAME_DATE = datetime.today().strftime('%Y-%m-%d')

# Load the trained model (Make sure the path is correct)
model = joblib.load('enhanced_fantasy_model.pkl')  # Update path as necessary


# Fetch DraftKings DFS slates for the given date
def fetch_dfs_slates(date=GAME_DATE):
    """Fetch DraftKings DFS slates for the given date"""
    url = f"https://api.sportsdata.io/api/nba/fantasy/json/DfsSlatesByDate/{date}?key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        slate_data = response.json()
        # Automatically fetch the "Main" slate
        main_slate = next((slate for slate in slate_data if slate["OperatorName"] == "Main"), None)
        return main_slate
    return None


# Fetch daily NBA player projections for the main slate
def fetch_player_stats(date=GAME_DATE, selected_slate=None):
    """Fetch daily NBA player projections for players in the main slate"""
    if not selected_slate:
        st.error("No DraftKings main slate selected!")
        return pd.DataFrame()  # Return empty dataframe if no slate selected

    # Get the SlateID of the main slate
    slate_id = selected_slate.get("SlateID", None)
    if not slate_id:
        st.error("Main slate has no SlateID!")
        return pd.DataFrame()  # Return empty dataframe if SlateID is not found

    # Get the games in the main slate (SlateID)
    main_slate_games = selected_slate.get("DfsSlateGames", [])
    if not main_slate_games:
        st.error("No games found in the main slate!")
        return pd.DataFrame()  # Return empty dataframe if no games in main slate

    # Extract all GameIDs for the main slate games
    main_slate_game_ids = [game["GameID"] for game in main_slate_games]

    # Construct the URL to fetch player projections
    url = f"https://api.sportsdata.io/api/nba/fantasy/json/PlayerGameProjectionStatsByDate/{date}?key={API_KEY}"

    response = requests.get(url)

    if response.status_code == 200:
        df = pd.DataFrame(response.json())

        # Filter players by GameID to only include players in the main slate games
        df = df[df["GameID"].isin(main_slate_game_ids)]  # Only include players from the main slate games

        # Rename columns for consistency
        df.rename(columns={"PlayerID": "player_id", "Position": "position"}, inplace=True)
        df["name"] = df["Name"].apply(lambda x: x.strip())  # Normalize player names
        df["team"] = df["Team"]  # Use the 'Team' column from the API directly

        # Clean up data by removing unnecessary columns
        df.drop(columns=["TeamID", "InjuryStatus"], inplace=True, errors="ignore")  # Remove TeamID and InjuryStatus
        df = df[df["Minutes"] != 0]  # Exclude players with zero minutes
        df.fillna(0, inplace=True)  # Replace NaNs with 0

        # Exclude specific players by name
        excluded_players = ["Damion Baugh", "Jahlil Okafor", "Patrick Baldwin Jr.", "Alex Len", "Skal Labissiere", "Markelle Fultz", "Jaylen Nowell", "Lonnie Walker IV", "Killian Hayes", "Jahmir Young", "Jalen Hood-Schifino", "Lamar Stevens", "Javonte Green", "Jalen McDaniels", "Bones Hyland", "Jalen Crutcher"]
        df = df[~df["name"].isin(excluded_players)]  # Exclude players based on their names

        return df
    else:
        st.error("Error fetching player projections.")
        return pd.DataFrame()  # Return empty dataframe in case of error


# Fetch game odds
def fetch_game_odds(date=GAME_DATE):
    """Fetch NBA game odds"""
    url = f"https://api.sportsdata.io/api/nba/odds/json/GameOddsByDate/{date}?key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []


# Merge DraftKings salary data using PlayerID
def merge_dfs_salary(merged_data, selected_slate):
    """Merge salary information from DraftKings slate into player data using PlayerID"""
    slate_players_df = pd.DataFrame(selected_slate["DfsSlatePlayers"])

    slate_players_df = slate_players_df.rename(columns={
        "PlayerID": "player_id",
        "OperatorSalary": "salary",
        "OperatorPosition": "position",
        "OperatorPlayerName": "name"
    })[["player_id", "salary", "position", "name"]]  # Keep only necessary columns

    # Merge the salary data with the existing merged_data
    merged_data = pd.merge(merged_data, slate_players_df, on="player_id", how="left")

    # Rename 'name_y' and 'position_y' to 'name' and 'position' after merge if necessary
    if 'name_y' in merged_data.columns:
        merged_data["name"] = merged_data["name_y"]
        merged_data.drop("name_y", axis=1, inplace=True)

    if 'position_y' in merged_data.columns:
        merged_data["position"] = merged_data["position_y"]
        merged_data.drop("position_y", axis=1, inplace=True)

    # Ensure necessary columns are present after renaming
    required_columns = ['name', 'position', 'salary']
    missing_columns = [col for col in required_columns if col not in merged_data.columns]
    if missing_columns:
        st.error(f"Missing columns in merged_data: {', '.join(missing_columns)}")

    return merged_data


# Merge game odds into the player data
def merge_game_odds(game_odds, merged_data):
    """Merge game odds into the player data"""
    game_odds_dict = {}
    for game in game_odds:
        away_team = game["AwayTeamName"]
        home_team = game["HomeTeamName"]
        if "PregameOdds" in game and game["PregameOdds"]:
            spread = game["PregameOdds"][0].get("AwayPointSpread", 0)
            total_points = game["PregameOdds"][0].get("OverUnder", 0)
        else:
            spread = 0
            total_points = 0

        game_odds_dict[away_team] = {"spread": spread, "total_points": total_points}
        game_odds_dict[home_team] = {"spread": game["PregameOdds"][0].get("HomePointSpread", 0),
                                     "total_points": total_points}

    merged_data["spread"] = merged_data["team"].apply(lambda x: game_odds_dict.get(x, {}).get("spread", 0))
    merged_data["total_points"] = merged_data["team"].apply(lambda x: game_odds_dict.get(x, {}).get("total_points", 0))

    return merged_data


from sklearn.preprocessing import StandardScaler


def calculate_additional_features(merged_data):
    """Calculate and add UsageRate, team pace, offensive rating, defensive rating, and interaction features"""

    # Calculate UsageRate
    merged_data["UsageRate"] = (merged_data["FieldGoalsAttempted"] + 0.44 * merged_data["FreeThrowsAttempted"] +
                                merged_data["Turnovers"]) / (merged_data["Minutes"] + 1)

    # Aggregate team stats (sum of relevant stats for all players on the team)
    team_stats = merged_data.groupby("team").agg({
        "FieldGoalsAttempted": "sum",
        "FreeThrowsAttempted": "sum",
        "OffensiveRebounds": "sum",
        "Turnovers": "sum",
        "Minutes": "sum"
    }).reset_index()

    # Calculate team pace using the aggregated team stats
    team_stats["team_pace"] = (48 * (
            team_stats["FieldGoalsAttempted"] + 0.44 * team_stats["FreeThrowsAttempted"] - team_stats[
        "OffensiveRebounds"] + team_stats["Turnovers"])) / (team_stats["Minutes"] / 5)

    # Merge team pace back to the player-level data (so each player on the same team has the same team_pace)
    merged_data = pd.merge(merged_data, team_stats[["team", "team_pace"]], on="team", how="left")

    # If 'team_pace_y' exists (because of duplicate columns), rename it to 'team_pace'
    if "team_pace_y" in merged_data.columns:
        merged_data.rename(columns={"team_pace_y": "team_pace"}, inplace=True)

    # Calculate offensive rating
    merged_data["offensive_rating"] = (100 * merged_data["Points"]) / (
            merged_data["FieldGoalsAttempted"] + 0.44 * merged_data["FreeThrowsAttempted"])

    # Calculate defensive rating based on available stats like steals, blocks, and defensive efficiency
    merged_data["defensive_rating"] = (merged_data["Steals"] + merged_data["BlockedShots"]) / merged_data["Minutes"]

    # Calculate points per minute
    merged_data["points_per_minute"] = merged_data["Points"] / merged_data["Minutes"]

    # Scale the spread and total points (important for their impact on model learning)
    scaler = StandardScaler()

    # Scale spread and total_points
    merged_data['scaled_spread'] = scaler.fit_transform(merged_data[['spread']])
    merged_data['scaled_total_points'] = scaler.fit_transform(merged_data[['total_points']])

    # Create interaction features
    merged_data['spread_team_pace'] = merged_data['scaled_spread'] * merged_data['team_pace']
    merged_data['spread_total_points'] = merged_data['scaled_spread'] * merged_data['scaled_total_points']

    return merged_data


def generate_projections(merged_data, selected_slate):
    """Generate projections using the enhanced model and add relevant columns"""
    if merged_data is None:
        return None

    # If more than 200 players, limit to top 190 by FantasyPoints
    if len(merged_data) > 200:
        merged_data = merged_data.nlargest(180, 'FantasyPoints')  # Take the top 190 players based on FantasyPoints

    # Calculate the necessary additional features
    merged_data = calculate_additional_features(merged_data)

    # Ensure the data has the features the model expects
    model_features = model.get_booster().feature_names

    # Check if the necessary columns for the model are present
    missing_columns = [col for col in model_features if col not in merged_data.columns]
    if missing_columns:
        st.error(f"Missing columns for model: {', '.join(missing_columns)}")
        return None

    # Reorder the merged data to match the model's feature order
    merged_data_for_model = merged_data[model_features]


    # Generate projections using the model
    projections = model.predict(merged_data_for_model)


    # Use .loc[] to avoid SettingWithCopyWarning
    merged_data.loc[:, "FantasyPoints"] = projections

    # Store necessary columns (e.g., 'name', 'position', 'salary') separately before reordering
    necessary_columns = ['name', 'position', 'salary']

    # Now, keep the necessary columns for the final result
    merged_data[necessary_columns] = merged_data[necessary_columns]

    # Combine data after projections
    combined_data = merged_data[['name', 'position', 'salary', 'FantasyPoints']]

    # Save the combined data in session state for use in later steps
    st.session_state.combined_data = combined_data

    return merged_data



from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD
import time
import pandas as pd

def optimize_lineup(players, salary_cap=50000, roster_size=8):
    st.write("Starting optimization...")
    # Initialize the optimization problem
    problem = LpProblem("DraftKings_Lineup", LpMaximize)

    # Initialize binary variables for each player
    player_vars = {row["name"]: LpVariable(row["name"], 0, 1, cat="Binary") for _, row in players.iterrows()}

    # Objective: Maximize FantasyPoints
    problem += lpSum(row["FantasyPoints"] * player_vars[row["name"]] for _, row in players.iterrows())

    # Add constraints: Total salary must be <= salary_cap
    problem += lpSum(row["salary"] * player_vars[row["name"]] for _, row in players.iterrows()) <= salary_cap

    # Roster size constraint: Ensure exactly 8 players are selected
    problem += lpSum(player_vars[row["name"]] for _, row in players.iterrows()) == roster_size

    # Define position eligibility mapping (G, F, UTIL are special roster spots)
    eligible_positions = {
        "PG": ["PG", "PG/SG", "SG/PG", "PG/SF", "SF/PG"],
        "SG": ["SG", "PG/SG", "SG/SF", "SG/PF", "PG/SG"],
        "SF": ["SF", "SG/SF", "SF/PF", "PF/SF", "PG/SF"],
        "PF": ["PF", "SF/PF", "PF/C", "C/PF", "SF/PF"],
        "C": ["C", "PF/C", "C/PF"]
    }

    # Roster spot eligibility (G, F, UTIL as special designations)
    eligible_roster_spots = {
        "G": ["PG", "SG", "PG/SG", "SG/SF", "PG/SF"],  # Includes PG/SF for G
        "F": ["SF", "PF", "SF/PF", "PF", "SF/PF", "PF/C", "C/PF"],  # Includes SF/PF and PF/C for F
        "UTIL": ["PG", "SG", "SF", "PF", "C", "PG/SG", "SG/SF", "SF/PF", "PF/C", "PG/SF", "SF/PG"]  # Includes all dual-position players for UTIL
    }

    # Add position constraints (at least 1 for each position)
    for position, eligible_values in eligible_positions.items():
        count = lpSum(player_vars[row["name"]] for _, row in players.iterrows() if any(val in row["position"] for val in eligible_values))
        problem += count >= 1  # At least 1 player for each primary position (PG, SG, SF, PF, C)

    # Add roster spot constraints (G, F, UTIL)
    for roster_spot, eligible_values in eligible_roster_spots.items():
        count = lpSum(player_vars[row["name"]] for _, row in players.iterrows() if any(val in row["position"] for val in eligible_values))
        problem += count >= 1  # Ensure there is 1 player for each roster spot (G, F, UTIL)

    # Solve the optimization problem
    problem.solve(PULP_CBC_CMD(msg=True, presolve=True))

    # Check solver status
    if problem.status != 1:
        st.error(f"Solver returned status: {problem.status}. The problem might be infeasible.")
        return None

    # Return results
    if problem.status == 1:  # Optimal solution found
        lineup = players[players["name"].isin([key for key, var in player_vars.items() if var.value() == 1])]

        # Filter to only show the desired columns in the optimized lineup
        optimized_lineup_filtered = lineup[['name', 'salary', 'position', 'FantasyPoints']]

        return optimized_lineup_filtered
    else:
        st.error(f"No optimal solution found. Solver returned status: {problem.status}")
        return None


# STREAMLIT APP

st.set_page_config(

    page_title="NBA Lineup Optimizer",  # Replace with your app's name
    page_icon="melo.png" # Path to your icon image (image must be in the same folder as the script)

)
# Add a date picker to allow the user to select the game date
st.sidebar.write("### Select Game Date")
game_date = st.sidebar.date_input(
    "Pick a date for the game", datetime.today().date()  # Default to today's date
)

# Format the selected date to match the API format (YYYY-MM-DD)
GAME_DATE = game_date.strftime('%Y-%m-%d')  # Convert to 'YYYY-MM-DD'

# Now GAME_DATE is set dynamically based on the selected date
st.write(f"Selected Game Date: {GAME_DATE}")

# Automatically fetch the "Main" DraftKings slate based on selected GAME_DATE
selected_slate = fetch_dfs_slates(date=GAME_DATE)

if not selected_slate:
    st.error(f"No DraftKings slate found for {GAME_DATE}.")
else:
    st.write(f"Selected Slate for {GAME_DATE}: {selected_slate['OperatorName']} for {selected_slate['OperatorDay']}")

    # Load Data button
    if st.sidebar.button("Step 1: Load Data"):
        if selected_slate and 'SlateID' in selected_slate:
            players = fetch_player_stats(date=GAME_DATE, selected_slate=selected_slate)

            if not players.empty:
                merged_data = merge_dfs_salary(players, selected_slate)

                game_odds = fetch_game_odds(date=GAME_DATE)
                merged_data = merge_game_odds(game_odds, merged_data)

                # Calculate and merge additional features (UsageRate, offensive_rating, defensive_rating, team_pace)
                merged_data = calculate_additional_features(merged_data)

                st.write("### Loaded Data")
                st.dataframe(merged_data)

                st.session_state.merged_data = merged_data

    # Generate Projections button
    if st.sidebar.button("Step 2: Generate Projections"):
        if "merged_data" in st.session_state:
            merged_data = st.session_state.merged_data

            # Generate projections using the model (this is already in your function)
            merged_data = generate_projections(merged_data, selected_slate)

            # After projections, ensure that only the top 190 players are kept based on FantasyPoints
            if len(merged_data) > 180:
                merged_data = merged_data.nlargest(180, 'FantasyPoints')  # Limit to top 190 based on FantasyPoints
                st.write(f"Top 190 players by FantasyPoints selected: {len(merged_data)} players")

            # Store the filtered data in session state for the optimizer
            st.session_state.filtered_data = merged_data

            st.write("### Projections Generated")
            st.dataframe(merged_data[['name', 'salary', 'position', 'FantasyPoints']])

        else:
            st.error("Merged data not found. Please load data first.")

    # Lineup Optimizer button
    with st.expander("Lineup Optimizer"):
        st.write("Use this section to optimize your fantasy lineup.")

        if st.button("Optimize Lineup"):
            # Check that filtered data is available in session state
            if "filtered_data" not in st.session_state:
                st.error("Filtered player data is not available. Please generate projections first.")
            else:
                players = st.session_state.filtered_data  # Get the filtered data from session state

                st.write("### Filtered Player Pool")
                st.dataframe(players[['name', 'salary', 'position', 'FantasyPoints']])  # Show filtered player pool

                # Run the optimizer function to get the optimal lineup
                optimized_lineup = optimize_lineup(players, salary_cap=50000, roster_size=8)

                # Check if the optimizer found a valid lineup
                if optimized_lineup is not None and not optimized_lineup.empty:
                    total_salary = optimized_lineup["salary"].sum()
                    total_fantasy_points = optimized_lineup["FantasyPoints"].sum()

                    # Display the optimized lineup with only the relevant columns
                    st.write("### Optimized Lineup")
                    st.dataframe(optimized_lineup)

                    st.write(f"**Total Salary**: ${total_salary}")
                    st.write(f"**Total Fantasy Points**: {total_fantasy_points}")
                else:
                    st.error("No optimal lineup found. Please adjust your data or constraints.")




























































































































































































