import pandas as pd
import random

def generate_txt_files(csv_file, seed_name):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Initialize output filenames
    genuine_file = f"{seed_name}_genuine_pairs.txt"
    impostor_file = f"{seed_name}_impostor_pairs.txt"

    # Open files to write
    with open(genuine_file, "w") as genuine_f, open(impostor_file, "w") as impostor_f:
        for _, row in df.iterrows():
            # Generate random IDs
            id1 = random.randint(1000, 9999)
            id2 = random.randint(1000, 9999)

            # Write to the appropriate file
            if row['IsGenuine']:
                # Prepare the line
                line = f"{id1} img_{row['Image1_ID']}.png {id1} img_{row['Image2_ID']}.png\n"
                genuine_f.write(line)
            else:
                # Prepare the line
                line = f"{id1} img_{row['Image1_ID']}.png {id2} img_{row['Image2_ID']}.png\n"
                impostor_f.write(line)

# Example usage
# Replace 'your_csv_file.csv' with your actual CSV file name
# Replace 'seed_name' with your desired seed name
generate_txt_files('exp/datasets/LFW_benchmark_pairs_mapping.csv', 'LFW_benchmark')