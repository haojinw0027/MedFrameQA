"""VQA Process Module."""
import os
from utils import load_config, load_results, save_results
from openai_inference import OpenAIInference
import json

MAXIMUM_FRAME_NUM = 5


class VQAProcess:
    """VQA Process Class."""

    def __init__(self, system, organ, keyword, video_id, max_frame_num):
        """Initializes the VQAProcess object.

        Args:
            system (str): The system name.
            organ (str): The target organ or body part.
            keyword (str): A keyword associated with the video.
            video_id (str): Unique identifier for the video.
            max_frame_num (int): Maximum number of consecutive frames to combine.

        Attributes:
            pairs_file (str): The path to the pairs JSON file.
            vqa_file (str): The path to the VQA JSON file.
            openai_interface: An instance for interfacing with OpenAI inference.
            prompt_path (str): The path to the prompt configuration YAML file.
            prompt (dict): The loaded configuration from the prompt file.
            temperature (float): The temperature parameter for the OpenAI API.
            max_tokens (int): The maximum number of tokens for the OpenAI API response.
            engine (str): The engine specified in the prompt configuration.
            max_attempts (int): The maximum number of attempts for OpenAI requests.
            vqa_system_prompt (str): The system prompt for the VQA process.
            vqa_prompt_template (str): The prompt template used for generating VQA questions.
        """
        self.vqa_process_status = True
        self.system = system
        self.organ = organ
        self.keyword = keyword
        self.video_id = video_id
        self.pairs_file = self._find_pairs_file()
        self.max_frame_num = min(MAXIMUM_FRAME_NUM, max_frame_num)
        self.vqa_file = self._find_vqa_file()
        self.openai_interface = OpenAIInference(config_path='../config/clients.yaml')
        self.prompt_path = '../config/prompt.yaml'
        self.prompt = load_config(self.prompt_path)
        self.relate_system_prompt = self.prompt["relate_system_prompt"]
        self.relate_VQA_template = self.prompt["relate_VQA_template"]
        self.vqa_system_prompt = self.prompt["VQA_system_prompt"]
        self.vqa_prompt_template = self.prompt["Multi_VQA_template"]

    def _find_pairs_file(self):
        """Finds the pairs file by constructing its path from the provided attributes.

        Constructs a folder path based on system, organ, keyword, and video_id. The path to the
        'pairs.json' file is then generated. If the file does not exist, a FileNotFoundError is raised.

        Raises:
            FileNotFoundError: If the pairs file is not found at the computed location.

        Returns:
            str: The full path to the pairs JSON file.
        """
        folder_path = os.path.join("../data", self.system, self.organ, self.keyword, self.video_id)
        file_path = os.path.join(folder_path, "pairs.json")
        if not os.path.exists(file_path):
            print(f"We cannot find the pairs file: {file_path}")
            self.vqa_process_status = False
        return file_path

    def _find_vqa_file(self):
        """Finds or creates the VQA file.

        Constructs the file path for the 'vqa.json' file based on system, organ, keyword, and video_id.

        If the file does not exist, the necessary directory is created to store the file.

        Returns:
            str: The full path to the VQA JSON file.
        """
        folder_path = os.path.join("../data", self.system, self.organ, self.keyword, self.video_id)
        file_path = os.path.join(folder_path, f"vqa_{self.max_frame_num}.json")
        if not os.path.exists(file_path) and self.vqa_process_status:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    def get_latest_checkpoint(self):
        """Retrieves the latest processed checkpoint from the VQA results.

        Attempts to load the VQA results from the vqa_file. It then iterates through the results to find the maximum frame number processed. If any exception occurs during loading, the error is printed and None is returned.

        Returns:
            tuple: A tuple containing:
                    - A list of previously processed results.
                    - The maximum frame number found.
                    If an exception occurs during loading, returns None.
        """
        max_frame = 0
        results = []
        try:
            results = load_results(self.vqa_file)
        except Exception as e:
            print("Error loading results:", e)
            return results, max_frame
        for record in results:
            if "frame" in record and record["frame"]:
                current_max = max(record["frame"])
                if current_max > max_frame:
                    max_frame = current_max
        print(f"Latest processed frame: {max_frame}")
        return results, max_frame

    def process_pairs(self):
        """Processes each pair from the pairs file to generate VQA responses.

        Reads the contents of the pairs JSON file and resumes processing from the latest checkpoint.

        For each item:
            - If the 'result' field exists and its value is "no", the frame is skipped.
            - If the 'result' field exists and its value is "yes", consecutive items (up to max_frame_num)
                with 'yes' are combined for processing.
            - Otherwise, processes the frame without a valid 'result' field.
        After processing each item or group of items, the updated VQA results are saved to the vqa_file.

        Returns:
            None
        """
        pairs = load_results(self.pairs_file)
        output_result, idx = self.get_latest_checkpoint()  # checkpoint
        while idx < len(pairs):
            item = pairs[idx]
            if "result" in item and item["result"].lower() == "no":
                print(f"Skip frame {idx+1}.")
                idx += 1
                continue
            if any(result['frames'][0] <= idx+1 <= result['frames'][-1] for result in output_result):
                print(f"Skip Processed frame {idx+1}.")
                idx += 1
                continue
            if "result" in item and item["result"].lower() == "yes":
                transcripts = [item.get("rephrased_description", "")]
                all_frame_idx = [idx+1]
                start_idx = idx
                idx += 1
                count = 1
                # Process consecutive yes frames
                while idx < len(pairs) and "result" in pairs[idx] and pairs[idx]["result"].lower() == "yes":
                    # Stop adding if maximum frame limit is reached
                    if count >= self.max_frame_num:
                        break
                    # Add new frame
                    transcripts.append(pairs[idx].get("rephrased_description", ""))
                    all_frame_idx.append(idx+1)
                    count += 1
                    # If only one frame, continue adding
                    if count == 1:
                        idx += 1
                        continue
                    # Verify relation between all current frames
                    combined = {f"transcript_{i+1}": t for i, t in enumerate(transcripts)}
                    print(f"Processing relation verification for frames {start_idx+1} to {idx+1}.")
                    related = self.verify_relation(all_frame_idx, combined)
                    if 'raw_response' in related:
                        print(f"Error verifying relation: {related['raw_response']}")
                        return
                    response = self.parse_related_vqa(related)
                    # If pairs_of_related_frames length is not 1, frames are not related, stop adding
                    if len(response['pairs_of_related_frames']) != 1:
                        # Remove last added frame
                        transcripts.pop()
                        all_frame_idx.pop()
                        count -= 1
                        break
                    idx += 1
                # Process collected related frames
                if count > 1:
                    combined = {f"transcript_{i+1}": t for i, t in enumerate(transcripts)}
                    print(f"Processing VQA for frames {start_idx+1} to {start_idx+count}.")
                    related = self.verify_relation(all_frame_idx, combined)
                    if 'raw_response' in related:
                        print(f"Error verifying relation: {related['raw_response']}")
                        return
                    response = self.parse_related_vqa(related)
                    for item in response['pairs_of_related_frames']:
                        frames = list()
                        for frame_idx in item['selected_transcripts']:
                            frames.append(response['frames'][frame_idx-1])
                        output_result.append(self.process_item(frames, item))
            else:
                print(f"Processing frame {idx+1} without valid 'result'.")
                idx += 1
            save_results(output_result, self.vqa_file)

    def verify_relation(self, all_frame_idx: list, transcript_items: dict):
        """Verify the relation between the transcript and the keyword.

        Args:
            all_frame_idx (list): The frame indices to verify.
            transcript_items (dict): The transcript items to verify.

        Returns:
            dict: The dictionary containing the response from the OpenAI API, including details such as
                    frame indices and the processed transcript.
        """
        prompt = self.relate_VQA_template.format(
            keyword=self.keyword,
            body_part=self.organ,
            caption=transcript_items
        )
        response = self.openai_interface.run(
            max_tokens=10000,
            prompt=prompt,
            system_prompt=self.relate_system_prompt
        )
        try:
            response = json.loads(response)
        except Exception as e:
            response = dict()
            response['raw_response'] = response
            print(f"Error parsing response: {e}")
        response['frames'] = all_frame_idx
        response['transcripts'] = transcript_items
        return response

    def parse_related_vqa(self, related_vqa: dict):
        """Parse the related VQA response.

        Args:
            related_vqa (dict): The related VQA response.
        """
        response = dict()
        response['frames'] = related_vqa['frames']
        response['pairs_of_related_frames'] = related_vqa['pairs_of_related_frames']
        for i, related_vqa_frame in enumerate(related_vqa['pairs_of_related_frames']):
            image_index = related_vqa_frame['selected_transcripts']
            res = dict()
            for idx in image_index:
                res[f'transcript_{idx}'] = related_vqa['transcripts'][f'transcript_{idx}']
            response['pairs_of_related_frames'][i]['transcripts'] = res
        return response

    def process_item(self, all_frame_idx, item):
        """Processes a single or combined set of items to generate a VQA answer.

        Formats the prompt using the provided organ, keyword, and transcript(s). Calls the OpenAI API
        with the generated prompt and additional parameters such as temperature, max_tokens, engine, and max_attempts.
        The API response is then converted to a dictionary and augmented with frame and transcript information.

        Args:
            all_frame_idx (list or None): A list of frame indices associated with the current item(s)
                                            or None if not applicable.
            item (dict): The transcript data or text for the VQA processing.

        Returns:
            dict: The dictionary containing the response from the OpenAI API, including details such as
                    frame indices and the processed transcript.
        """
        transcripts = item['transcripts']
        if "transcript_1" not in transcripts:
            transcripts = {f"transcript_{i+1}": transcripts[t] for i, t in enumerate(transcripts)}
        prompt = self.vqa_prompt_template.format(
            body_part=self.organ,
            keyword=self.keyword,
            transcript=transcripts
        )
        response = self.openai_interface.run(
            max_tokens=10000,
            prompt=prompt,
            system_prompt=self.vqa_system_prompt
        )
        try:
            response = json.loads(response)
        except Exception as e:
            response = dict()
            response['raw_response'] = response
            print(f"Error parsing response: {e}")
        response['frames'] = all_frame_idx
        response['transcripts'] = transcripts
        response['related_reason'] = item['related_reason']
        return response


if __name__ == "__main__":
    """Main function to execute the VQA process."""
    inf = 99999999
    vqa_processor = VQAProcess("central_nervous_system", "brain", "skull_radiograph", "ElDGBiKupyE", inf)
    vqa_processor.process_pairs()
