"""evaluate model performance on the benchmark dataset."""
import os
import json
import base64
import random
import argparse
import numpy as np
from tqdm import tqdm
from openai_inference import OpenAIInference
from utils import load_config, load_results, save_results
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

def image_to_data_url(image_path: str) -> str:
    """Convert an image file to a data URL.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The data URL.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lstrip(".").lower()
        return f"data:image/{ext};base64,{encoded_string}"


def add_letter(options: list) -> list:
    """Add letters to the options for better readability.

    Args:
        options (list): List of options to be formatted.

    Returns:
        list: List of formatted options with letters.

    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [f"{letters[i]}. {option}" for i, option in enumerate(options)]

def parse_multi_choice_response(response: str, all_choices: list, index2ans: dict, question_id: str) -> str:
    """Parse the prediction from the generated response.

    Args:
        response (str): The generated response.
        all_choices (list): List of all choices.
        index2ans (dict): Dictionary mapping index to answer.

    Returns:
        str: The predicted answer.
    """
    # print(question_id)
    
    # Check if there is a boxed{LETTER} format answer
    boxed_pattern = r'boxed\{([A-Z])\}'
    boxed_matches = re.findall(boxed_pattern, response)
    
    if boxed_matches:
        for match in boxed_matches:
            # Check if the matched letter is in the options
            if match in all_choices:
                return match
    answer_str = ""
    # Find various possible answer markers
    answer_markers = [
        "Answer:",
        "Answer*",
        "Final Answer",
        "final answer"
    ]
    # Iterate through all possible answer markers
    for marker in answer_markers:
        last_answer_pos = response.rfind(marker)
        if last_answer_pos != -1:
            # Extract the string after the marker as the answer
            answer_str = response[last_answer_pos + len(marker):].strip()
            break

    if answer_str != "":
        matching_options = [option for option in all_choices if option in answer_str]

        # If a unique match is found, return that option
        if len(matching_options) >= 1:
            # Find all matching option positions in answer_str
            option_positions = [(option, answer_str.find(option)) for option in matching_options]
            # Filter out options not found (position -1)
            valid_options = [(option, pos) for option, pos in option_positions if pos != -1]
            # If there are valid options, return the one that appears first
            if valid_options:
                return min(valid_options, key=lambda x: x[1])[0]
            return matching_options[0]  # If no position info found, return first matching option

    if isinstance(response, str):
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
        response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)
    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.
    print(question_id, candidates)
    if len(candidates) == 0:  # still not get answer, none
        pred_index = ""
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
            else:
                for can in candidates:
                    index = response.rfind(f"{can}.")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def api_eval(system_prompt: str, prompt: str, image_paths: list, model_name: str, choices: list, index2ans: dict, question_id: str, max_tokens: int = 10000) -> dict:
    """Evaluate the model using OpenAI API.

    Args:
        system_prompt (str): The system prompt to be used.
        prompt (str): The prompt to be used for the evaluation.
        image_paths (list): List of image paths to be used for the evaluation.
        model_name (str): The name of the model to be used.
        choices (list): List of choices to be used for the evaluation.
        index2ans (dict): Dictionary mapping index to answer.

    Returns:
        dict: The response from the model.

    """
    result = {}
    openai_interface = OpenAIInference(config_path='../config/clients.yaml')
    response = openai_interface.run(system_prompt, prompt, image_paths=image_paths, engine=model_name, max_tokens=max_tokens)
    result["cot_reasoning"] = response
    result["answer"] = parse_multi_choice_response(response, choices, index2ans, question_id)
    return result

def evaluate(input_file: str, output_dir: str, model_name: str, num_q: int, system_prompt: str, prompt_template: str, max_tokens: int = 10000) -> None:
    """Evaluate the model performance on the benchmark dataset.

    Args:
        input_file (str): Path to the input JSON file containing benchmark data.
        output_dir (str): Directory to save the results.
        model_name (str): Name of the model to be used for evaluation.
        num_q (int): Number of questions to be evaluated.
        system_prompt (str): System prompt to be used for the evaluation.
        prompt_template (str): Prompt template to be used for the evaluation.
    """
    try:
        with open(input_file, "r") as f:
            benchmark_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON from {input_file}: {e}")
    if num_q != -1:
        benchmark_data = benchmark_data[:min(num_q, len(benchmark_data))]
    results = load_results(output_dir)

    def process_item(data) -> dict:
        """Process each item in the benchmark data.

        Args:
            data (dict): The data item to be processed.

        Returns:
            dict: The processed result.
        """
        question_id = data["question_id"]
        question = data["question"]
        options = add_letter(data["options"])
        answer = data["correct_answer"]
        image_path = data["image_path"]
        keyword = data["keyword"]
        formatted_system_prompt = system_prompt.format(keyword=keyword)
        formatted_prompt = prompt_template.format(question=question, options=options)
        choices = [option.split(". ")[0] for option in options]
        index2ans = {}
        for i, opt in enumerate(options):
            key = chr(65+i)
            value = opt.split('. ')[1]
            index2ans[key] = value
        print(question_id)
        res = api_eval(formatted_system_prompt, formatted_prompt, image_path, model_name, choices, index2ans, question_id, max_tokens)
        res["question_id"] = question_id
        res["correct_answer"] = answer
        return res

    existing_ids = {res["question_id"] for res in results}
    remaining_data = [data for data in benchmark_data if data["question_id"] not in existing_ids]
    futures = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for data in remaining_data:
            futures.append(executor.submit(process_item, data))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
                if result["cot_reasoning"] == "":
                    continue
                results.append(result)
                save_results(results, output_dir)
            except Exception as e:
                print(f"Error processing item: {e}")


def compute_accuracy(result_json_path: str, summary_path: str) -> None:
    """Compute the accuracy of the model's predictions.

    Args:
        result_json_path (str): Path to the JSON file containing the results.
        summary_path (str): Path to the JSON file to save the accuracy summary.

    """
    with open(result_json_path, "r") as f:
        results = json.load(f)
    total = len(results)
    if total == 0:
        print("No results found in result.json.")
        return
    correct = sum(1 for res in results if res.get("raw_response", []) == [] and res.get("answer") == res["correct_answer"])
    acc = correct / total
    accuracy_data = {
        "accuracy": f"{acc:.2%}",
        "correct": correct,
        "total": total
    }
    with open(summary_path, "w") as f:
        json.dump(accuracy_data, f, indent=4)
    print(f"Accuracy data saved to {summary_path}")


if __name__ == "__main__":
    """Main function to run the evaluation process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str, default="../eval/wrong_final.json")
    parser.add_argument("--output_dir", "-o", type=str, default="../eval")
    parser.add_argument("--model_name", "-m", type=str, default="gpt-4o",
                        choices=["gpt-4o", "o1", "gpt-4-turbo-V", "gpt-4o-mini", "Qwen2.5-VL-72B-Instruct", "QVQ-72B-Preview",\
                                 "Gemini-2.5-Flash-Preview", "Claude-3-7-sonnet", "o3", "o4-mini"])
    parser.add_argument("--num_q", "-n", type=int, default=-1)
    parser.add_argument("--max_tokens", "-t", type=int, default=10000)
    args = parser.parse_args()
    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, 'result.json')
    summary_path = os.path.join(output_dir, 'summary.json')
    prompt = load_config("../config/prompt.yaml")
    if args.model_name == "Gemini-2.5-Flash-Preview":
        system_prompt = prompt["gemini_system_prompt"]
    elif args.model_name == "QVQ-72B-Preview":
        system_prompt = prompt["qvq_system_prompt"]
    else:
        system_prompt = prompt["eval_system_prompt"]
    prompt_template = prompt["eval_prompt_template"]
    evaluate(args.input_file, result_path, args.model_name, args.num_q, system_prompt, prompt_template, args.max_tokens)
    compute_accuracy(result_path, summary_path)
