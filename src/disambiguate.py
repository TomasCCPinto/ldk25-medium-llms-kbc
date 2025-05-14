import json
import argparse
import requests

def disambiguation_baseline(item):
    """
    Resolve ambiguous entities by converting them to Wikidata IDs.
    """
    try:
        # If item can be converted to an integer, return it directly
        return int(item)
    except ValueError:
        # If not, proceed with the Wikidata search
        try:
            url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
            data = requests.get(url).json()
            # Return the first id (Could upgrade this in the future)
            return data['search'][0]['id']
        except:
            return item

def disambiguate_jsonl_file(input_file_path, output_file_path):
    """
    Read a JSONL file, resolve ambiguous entities in the ObjectEntitiesID field, and write the disambiguated data to a new file.
    """
    count =0
    with open(input_file_path, "r", encoding="utf-8") as infile, \
         open(output_file_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            row = json.loads(line)
            # Apply disambiguation to each entity in the ObjectEntitiesID list
            if "ObjectEntitiesID" in row:
                disambiguated_entities = [disambiguation_baseline(item) for item in row["ObjectEntitiesID"]]
                # Update the row with the disambiguated entities
                row["ObjectEntitiesID"] = disambiguated_entities
            # Write the disambiguated row to the output file
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(count)
            count=count+1

def clean_jsonl_line(line):
    try:
        # Parse the JSON line
        data = json.loads(line)
        
        # Clean the ObjectEntitiesID field if it exists and is a list with one string element
        if 'ObjectEntitiesID' in data and isinstance(data['ObjectEntitiesID'], list) and len(data['ObjectEntitiesID']) > 0:
            if isinstance(data['ObjectEntitiesID'][0], str):
                # Remove the unwanted characters
                cleaned = data['ObjectEntitiesID'][0].replace('[\"', '').replace('\"]', '').replace('\"', '')
                data['ObjectEntitiesID'] = [cleaned] if cleaned else []
        
        # Return the cleaned line as a JSON string
        return json.dumps(data, ensure_ascii=False)
    except json.JSONDecodeError:
        # Return the original line if it's not valid JSON
        return line.strip()

def clean_jsonl_file(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        for line in infile:
            cleaned_line = clean_jsonl_line(line)
            outfile.write(cleaned_line + '\n')


def main():
    
    parser = argparse.ArgumentParser(description="Disambiguate the ObjectEntitiesID field in a JSONL file.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input JSONL file (required)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output JSONL file (required)"
    )

    args = parser.parse_args()

    # Disambiguate the input file and save the result to the output file
    disambiguate_jsonl_file(args.input, args.output)
    print(f"Disambiguated data saved to {args.output}")
    

if __name__ == "__main__":
    main()