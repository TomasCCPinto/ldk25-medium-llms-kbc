import json
import argparse

def clean_object_entities(obj_entities):
    """
    Clean the ObjectEntitiesID field to remove extra characters and format it as a proper list.
    """
    if isinstance(obj_entities, list):
        # If the field is already a list, process each item
        cleaned_list = []
        for item in obj_entities:
            # Remove extra quotes and brackets
            if isinstance(item, str):
                item = item.strip('[]"')
                # Split by comma if there are multiple items in a single string
                if "," in item:
                    cleaned_list.extend([i.strip(' "') for i in item.split(",")])
                else:
                    cleaned_list.append(item.strip(' "'))
        return cleaned_list
    return obj_entities

def clean_jsonl_file(input_file_path, output_file_path):
    """
    Read a JSONL file, clean the ObjectEntitiesID field, and write the cleaned data to a new file.
    """
    with open(input_file_path, "r", encoding="utf-8") as infile, \
         open(output_file_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            row = json.loads(line)
            # Clean the ObjectEntitiesID field
            row["ObjectEntitiesID"] = clean_object_entities(row["ObjectEntitiesID"])
            # Write the cleaned row to the output file
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Clean the ObjectEntitiesID field in a JSONL file.")
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

    # Clean the input file and save the result to the output file
    clean_jsonl_file(args.input, args.output)
    print(f"Cleaned data saved to {args.output}")

if __name__ == "__main__":
    main()