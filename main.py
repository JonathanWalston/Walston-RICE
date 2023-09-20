import data_collection
import car_calculation
import car_stats


def main():
    # Collect data
    print("Starting data collection...")
    try:
        data_collection.collect_data()
        print("Data collection completed successfully.")
    except Exception as e:
        print(f"Data collection failed with error: {e}")
        return

    # Calculate CAR
    print("Starting CAR calculation...")
    try:
        car_calculation.calculate_all_cars()
        print("CAR calculation completed successfully.")
    except Exception as e:
        print(f"CAR calculation failed with error: {e}")
        return

    # Calculate Statistics
    print("Starting statistics calculation...")
    try:
        car_stats.calculate_stats()
        print("Statistics calculation completed successfully.")
    except Exception as e:
        print(f"Statistics calculation failed with error: {e}")
        return

    print("All tasks completed successfully.")


if __name__ == '__main__':
    main()
