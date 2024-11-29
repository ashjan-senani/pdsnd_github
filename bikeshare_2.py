import time
import pandas as pd
import numpy as np

# Refactored CITY_FILE_PATHS to be more descriptive
CITY_FILE_PATHS = { 'chicago': 'chicago.csv',
                    'new york city': 'new_york_city.csv',
                    'washington': 'washington.csv' }

def get_filters():
    """
    Asks the user to specify a city, month, and day to analyze.
    
    Returns:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    """
    print('Hello! Let\'s explore some US bikeshare data!')
    
    # Get user input for city, ensuring the input is valid
    city = input("Please enter a city (chicago, new york city, washington): ").lower()
    while city not in CITY_FILE_PATHS:
        city = input("Invalid city. Please enter a valid city (chicago, new york city, washington): ").lower()
    
    # Get user input for month
    month = input("Please enter a month (all, january, february, ... , june): ").lower()
    
    # Get user input for day of the week
    day = input("Please enter a day (all, monday, tuesday, ... sunday): ").lower()
    
    print('-' * 40)
    return city, month, day

def load_data(city, month, day):
    """
    Loads data for the specified city and filters by month and day if applicable.

    Args:
        city (str): The name of the city to analyze ('chicago', 'new york city', or 'washington').
        month (str): The month to filter by, or "all" to apply no month filter.
        day (str): The day of the week to filter by, or "all" to apply no day filter.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the filtered bikeshare data for the given city, month, and day.
    """
    # Use the dictionary to load the city data file
    df = pd.read_csv(CITY_FILE_PATHS[city])

    # Filter by month if needed (month filter logic can be added here)
    if month != 'all':
        df = df[df['month'] == month.title()]

    # Filter by day if needed (day filter logic can be added here)
    if day != 'all':
        df = df[df['day_of_week'] == day.title()]
    
    return df

def time_stats(df):
    """Displays statistics on the most frequent times of travel."""
    print('\nCalculating The Most Frequent Times of Travel...\n')
    start_time = time.time()

    # Simplified common statistics for the month, day, and start hour
    most_common_month = df['month'].mode()[0]
    most_common_day = df['day_of_week'].mode()[0]
    most_common_hour = df['hour'].mode()[0]

    print(f"Most Common Month: {most_common_month}")
    print(f"Most Common Day: {most_common_day}")
    print(f"Most Common Start Hour: {most_common_hour}")

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)

def station_stats(df):
    """Displays statistics on the most popular stations and trip."""
    print('\nCalculating The Most Popular Stations and Trip...\n')
    start_time = time.time()

    # Most common start station
    most_common_start_station = df['start_station'].mode()[0]

    # Most common end station
    most_common_end_station = df['end_station'].mode()[0]

    # Most frequent combination of start station and end station trip
    most_common_trip = df.groupby(['start_station', 'end_station']).size().idxmax()

    print(f"Most Common Start Station: {most_common_start_station}")
    print(f"Most Common End Station: {most_common_end_station}")
    print(f"Most Common Trip: {most_common_trip}")

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)

def trip_duration_stats(df):
    """Displays statistics on the total and average trip duration."""
    print('\nCalculating Trip Duration...\n')
    start_time = time.time()

    # Total and mean travel time
    total_travel_time = df['trip_duration'].sum()
    mean_travel_time = df['trip_duration'].mean()

    print(f"Total Travel Time: {total_travel_time}")
    print(f"Mean Travel Time: {mean_travel_time}")

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)

def user_stats(df):
    """Displays statistics on bikeshare users."""
    print('\nCalculating User Stats...\n')
    start_time = time.time()

    # User statistics (counts of user types and gender)
    user_types = df['user_type'].value_counts()
    gender_counts = df['gender'].value_counts() if 'gender' in df.columns else "Gender data not available"
    
    # Year of birth statistics (if applicable)
    if 'birth_year' in df.columns:
        earliest_birth_year = df['birth_year'].min()
        most_recent_birth_year = df['birth_year'].max()
        most_common_birth_year = df['birth_year'].mode()[0]
    else:
        earliest_birth_year = most_recent_birth_year = most_common_birth_year = "Data not available"

    print(f"User Types:\n{user_types}")
    print(f"Gender Counts:\n{gender_counts}")
    print(f"Earliest Birth Year: {earliest_birth_year}")
    print(f"Most Recent Birth Year: {most_recent_birth_year}")
    print(f"Most Common Birth Year: {most_common_birth_year}")

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)

def main():
    while True:
        city, month, day = get_filters()
        df = load_data(city, month, day)

        time_stats(df)
        station_stats(df)
        trip_duration_stats(df)
        user_stats(df)

        restart = input('\nWould you like to restart? Enter yes or no.\n')
        if restart.lower() != 'yes':
            break

if __name__ == "__main__":
    main()
