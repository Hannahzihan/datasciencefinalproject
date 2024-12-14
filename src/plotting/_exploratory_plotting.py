import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_numerical_distributions(df, numerical_features):
    """
    Plot distribution charts for numerical features using subplots.

    Parameters:
        df (pd.DataFrame): The data frame containing numerical variables.
        numerical_features (list): A list of column names for the numerical features.
    """
    # Number of numerical features
    num_features = len(numerical_features)
    
    # Define the number of rows and columns for subplots
    cols = 3  # Number of columns in the subplot grid
    rows = (num_features // cols) + (num_features % cols > 0)  # Calculate rows needed
    
    # Set up the figure and axis objects
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration
    
    # Plot each numerical feature
    for i, feature in enumerate(numerical_features):
        sns.histplot(df[feature], kde=True, bins=30, ax=axes[i], color='blue', edgecolor='black')
        axes[i].set_title(f"Distribution of {feature}", fontsize=12)
        axes[i].set_xlabel(feature, fontsize=10)
        axes[i].set_ylabel("Frequency", fontsize=10)
    
    # Remove any unused subplots
    for i in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[i])
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_categorical_distributions_and_boxplots(df, categorical_features, target_feature):
    """
    Plot subplots for categorical features showing both distribution (countplot) and 
    relationships with a continuous target variable (boxplot).

    Parameters:
        df (pd.DataFrame): DataFrame containing the categorical variables and target variable.
        categorical_features (list): List of column names for the categorical variables.
        target_feature (str): Name of the continuous target feature for the boxplot.
    """
    # Define the number of rows and columns for the subplots
    num_features = len(categorical_features)
    cols = 2  # One column for countplot, one for boxplot
    rows = num_features  # Each row corresponds to one categorical feature
    
    # Create the subplot grid
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    # Plot each categorical feature
    for i, feature in enumerate(categorical_features):
        # Countplot on the left
        sns.countplot(data=df, x=feature, hue=feature ,palette="Set2", ax=axes[i, 0],legend=False)
        axes[i, 0].set_title(f"Distribution of {feature}", fontsize=12)
        axes[i, 0].set_xlabel(feature, fontsize=10)
        axes[i, 0].set_ylabel("Count", fontsize=10)
        axes[i, 0].tick_params(axis='x', rotation=45)

        # Add count labels on the countplot
        for patch in axes[i, 0].patches:
            height = patch.get_height()  # Get the height of the bar
            if height > 0:
                axes[i, 0].text(
                    patch.get_x() + patch.get_width() / 2,  # x-coordinate
                    height + 0.5,  # y-coordinate (slightly above the bar)
                    f'{int(height)}',  # Text to display (integer count)
                    ha='center', va='center', fontsize=9, color='black'
                )

        # Boxplot on the right
        sns.boxplot(x=feature, y=target_feature,hue=feature , data=df, palette="Set3", ax=axes[i, 1],legend=False)
        axes[i, 1].set_title(f"{target_feature} by {feature}", fontsize=12)
        axes[i, 1].set_xlabel(feature, fontsize=10)
        axes[i, 1].set_ylabel(target_feature, fontsize=10)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_month_hour_distribution(data):
    """
    Plot a month-hour distribution scatter plot where:
    - x-axis is the month,
    - y-axis is the hour,
    - point size represents the average rented bike count for each month-hour,
    - point color is determined by the season.

    Parameters:
        data (pd.DataFrame): DataFrame with columns 'Month', 'Hour', and 'Rented Bike Count'.
    """
    # Calculate average rented bike count for each month-hour combination
    grouped_data = data.groupby(['Month', 'Hour'], as_index=False)['Rented Bike Count'].mean()
    grouped_data['Rented Bike Count'] = grouped_data['Rented Bike Count'] / 10

    # Assign colors based on seasons
    def get_season_color(month):
        if month in [12, 1, 2]:
            return 'blue'  # Winter
        elif month in [3, 4, 5]:
            return 'pink'  # Spring
        elif month in [6, 7, 8]:
            return 'red'   # Summer
        elif month in [9, 10, 11]:
            return 'orange'  # Fall

    grouped_data['Color'] = grouped_data['Month'].apply(get_season_color)

    # Plotting
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        grouped_data['Month'],
        grouped_data['Hour'],
        s=grouped_data['Rented Bike Count'] * 10,  # Scale point size
        c=grouped_data['Color'],  # Point color
        alpha=0.7  # Make points slightly transparent
    )

    # Add labels and title
    plt.title("Month-Hour Distribution of Rented Bike Count", fontsize=16)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Hour", fontsize=12)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.yticks(range(0, 24))  # Hours from 0 to 23
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()