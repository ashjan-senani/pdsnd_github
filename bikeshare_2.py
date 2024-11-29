import time
import pandas as pd
import numpy as np

CITY_DATA = { 'chicago': 'chicago.csv',
              'new york city': 'new_york_city.csv',
              'washington': 'washington.csv' }

def get_filters():
    """
    Asks the user to specify a city, month, and day to analyze.

    The function prompts the user to input the city (chicago, new york city, washington),
    a month (all, january, february, ... , june), and a day of the week (all, monday, tuesday, ... sunday)
    for filtering the data.

    Returns:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    """
    print('Hello! Let\'s explore some US bikeshare data!')
    # get user input for city (chicago, new york city, washington)
    # get user input for month (all, january, february, ... , june)
    # get user input for day of week (all, monday, tuesday, ... sunday)
    print('-'*40)
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
    # Code to load data based on the city, month, and day
    return df



def time_stats(df):
    """
    Displays statistics on the most frequent times of travel.

    This function calculates and displays the most frequent month, day of the week, and start hour
    based on the provided bikeshare data.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing the filtered bikeshare data.

    Prints:
        - The most common month
        - The most common day of the week
        - The most common start hour
    """
    print('\nCalculating The Most Frequent Times of Travel...\n')
    start_time = time.time()

    # display the most common month
    # display the most common day of week
    # display the most common start hour

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)



def station_stats(df):
    """
    Displays statistics on the most popular stations and trip.

    This function calculates and displays statistics on the most commonly used start station,
    the most commonly used end station, and the most frequent combination of start and end stations.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing the filtered bikeshare data.

    Prints:
        - The most commonly used start station
        - The most commonly used end station
        - The most frequent combination of start and end station trip
    """
    print('\nCalculating The Most Popular Stations and Trip...\n')
    start_time = time.time()

    # display most commonly used start station
    # display most commonly used end station
    # display most frequent combination of start station and end station trip

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)



def trip_duration_stats(df):
    """
    Displays statistics on the total and average trip duration.

    This function calculates the total and mean travel time across all trips in the provided dataset.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing the filtered bikeshare data.

    Prints:
        - Total travel time across all trips
        - Mean travel time for all trips
    """
    print('\nCalculating Trip Duration...\n')
    start_time = time.time()

    # display total travel time
    # display mean travel time

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def user_stats(df):
    """
    Displays statistics on bikeshare users.

    This function calculates and displays statistics on the number of user types, gender, and the
    earliest, most recent, and most common year of birth for users.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing the filtered bikeshare data.

    Prints:
        - Counts of user types
        - Counts of gender
        - Earliest, most recent, and most common year of birth for users
    """
    print('\nCalculating User Stats...\n')
    start_time = time.time()

    # Display counts of user types
    # Display counts of gender
    # Display earliest, most recent, and most common year of birth

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
