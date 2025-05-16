"""Utils Functions."""
import os
import sys
import yaml
import base64
import json
import csv


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64 encoded image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_config(config_path: str) -> dict:
    """Loads the configuration from a YAML file.

    Args:
        config_path (str): The file path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration data.

    Exits:
        Exits the program if loading the configuration fails.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Failed to load configuration file: {e}")
        sys.exit(1)


def llm_result_to_dict(response: str) -> dict:
    """Convert a string representation of a dictionary to an actual dictionary.

    Args:
        str_dict (str): The string representation of the dictionary.
        ```json
        {
            "key1": "value1",
            "key2": "value2"
        }
        ```

    Returns:
        dict: The converted dictionary.
    """
    try:
        if "```" in response:
            lines = response.splitlines()
            if lines[0].strip().startswith("```") and lines[-1].strip().endswith("```"):
                response = "\n".join(lines[1:-1])
        output_dict = json.loads(response)
        return output_dict
    except Exception as e:
        print(f"Failed to convert string to dictionary: {e}")
        return {"raw_response": response}


def load_results(file_path: str):
    """Load existing results from a JSON file, or initialize if not present.

    Args:
        file_path (str): the location of the file.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return []  # Initialize if no file or empty/corrupted file


def save_results(results: dict, file_path: str):
    """Save the updated results to the JSON file.

    Args:
        results (dict): The result's dictionary.
        file_path (str): the location of the file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)


def get_video_id_from_csv(csv_dir: str) -> list:
    """Get the video ID from the CSV file.

    Args:
        csv_dir (str): The directory of the CSV file.

    Returns:
        list: [ {system, organ, keyword, video_id}]
    """
    try:
        rows = []
        with open(csv_dir, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rows.append(row)
        return rows
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []
