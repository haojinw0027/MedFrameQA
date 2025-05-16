"""Pre-processing the mapping with key-word extraction and double check."""
import os
import re
import json
from utils import load_config, load_results, save_results
from openai_inference import OpenAIInference
from tqdm import tqdm


class PairProcess:
    """Process the pair."""
    def __init__(self, system: str, organ: str, keyword: str, id: str):
        """Initialize the class.

        Args:
            system (str): The system.
            organ (str): The organ.
            keyword (str): The keyword.
            id (str): The id.
        """
        self.system = system
        self.organ = organ
        self.keyword = keyword
        self.id = id
        self.status = False
        self.video_invalid = False
        self.__parser__()
        self.openai_interface = OpenAIInference(config_path='../config/clients.yaml')
        self.prompt_path = '../config/prompt.yaml'
        self.prompt = load_config(self.prompt_path)
        self.pair_system_prompt = self.prompt["pair_system_prompt"]
        self.pair_prompt_template = self.prompt["pair_prompt_template"]

    def __parser__(self) -> None:
        """Parse the arguments."""
        self.path = f'../data/{self.system}/{self.organ}/{self.keyword}/{self.id}'
        self.pair_path = f'../data/{self.system}/{self.organ}/{self.keyword}/{self.id}/pairs.json'
        if os.path.exists(self.pair_path):
            var = load_results(self.pair_path)
            if var == []:
                self.status = True
        if not os.path.exists(os.path.join(self.path, "timestamps.txt")):
            self.video_invalid = True

    def parse_timestamps(self) -> list:
        """Parse the timestamps.txt file to extract frame numbers and corresponding times.

        Args:
            timestamps_file (str): The file path for the timestamps file.

        Return:
            A frame which consists of the frame componants.
        """
        frames = []
        with open(os.path.join(self.path, "timestamps.txt"), 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('frame:-1'):
                    match = re.match(r'frame:(\d+)\s+pts:\d+\s+pts_time:(\d+(?:\.\d+)?)', line)
                if match:
                    frame_num = int(match.group(1))
                    time = float(match.group(2))
                    frames.append((frame_num+1, time))  # start from 1
        return frames

    def match_text_to_frames(self, frames: list, bias_time: float = 0.0) -> list:
        """Match text to frames based on their timestamps.

        Args:
            audio2text_file (str): The file which converts the audio file into the string file.
            frames (list): The processed frames for later use.
            bias_time (float): The bias time for the text.

        Returns:
            A list that consist of raw text-image pair.
        """
        with open(os.path.join(self.path, f"transcription.json"), 'r', encoding='utf-8') as f:
            texts = json.load(f)
        results = []
        text_start_time = texts[0]["startTime"] if texts else 0
        text_end_time = texts[-1]["endTime"] if texts else 0
        frame_start_time = frames[0][1] if frames else 0
        frame_end_time = frames[-1][1] if frames else 0
        print(f"Text time range: {text_start_time:.2f}s - {text_end_time:.2f}s")
        print(f"Frame time range: {frame_start_time:.2f}s - {frame_end_time:.2f}s")
        for i in range(len(frames)):
            frame_index, start_time = frames[i]
            frame_start_time = start_time
            start_time = max(start_time - bias_time, text_start_time)
            if i < len(frames) - 1:
                end_time = min(frames[i+1][1] + bias_time, text_end_time)
                frame_end_time = frames[i+1][1]
            else:
                end_time = text_end_time
                frame_end_time = float('inf')
            print(f"Frame {frame_index} time range: {start_time:.2f}s - {end_time:.2f}s")
            print(f"Frame {frame_index} start time: {frame_start_time:.2f}s, end time: {frame_end_time:.2f}s")
            matching_texts = []
            for text in texts:
                if text["startTime"] > end_time or text["endTime"] < start_time:
                    continue
                matching_texts.append(text)
            results.append({
                "frame": frame_index,
                "startTime": start_time,
                "endTime": end_time,
                "frameStartTime": frame_start_time,
                "frameEndTime": frame_end_time,
                "sentences": matching_texts
            })
        return results

    def pair_process(self, pairs: list) -> list:
        """Pair process the frames.

        Args:
            frames (list): The processed frames for later use.

        Returns:
            A list that consist of raw text-image pair.
        """
        for frame in tqdm(pairs, desc="Pair processing frames"):
            results = load_results(self.pair_path)
            if any(result['frame'] == frame['frame'] for result in results):
                print(f"Frame {frame['frame']} already processed.")
                continue
            frame_path = os.path.join(self.path, f"keyframe_{frame['frame']}.jpg")
            if os.path.exists(frame_path):
                print(f"Found frame: {frame_path}")
            else:
                print(f"Frame not found: {frame_path}")
            system_prompt = self.pair_system_prompt.format(system=self.system, organ=self.organ, keyword=self.keyword)
            system_prompt = system_prompt.replace("{keyword}", self.keyword).replace("{body_part}", self.organ)
            prompt = self.pair_prompt_template.format(keyword=self.keyword, organ=self.organ, system=self.system, start_time=frame["startTime"], end_time=frame["endTime"], frame_start_time=frame["frameStartTime"], frame_end_time=frame["frameEndTime"], transcript_json_list=json.dumps(frame["sentences"]))
            response = self.openai_interface.run(system_prompt, prompt, image_paths=frame_path)
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {response}")
                results.append({'frame': frame['frame'], 'raw_response': response})
                save_results(results, self.pair_path)
                continue
            if response['result'] == 'yes':
                results.append({'frame': frame['frame'], 'result': response['result'], 'reason': response['reason'], 'transcripts': response['transcripts'], 'rephrased_description': response['rephrased_description']})
            else:
                results.append({'frame': frame['frame'], 'result': response['result'], 'reason': response['reason']})
            save_results(results, self.pair_path)

    def refine_results(self) -> None:
        """Refine the results."""
        results = load_results(self.pair_path)
        refined_results = []
        for result in results:
            if 'raw_response' in result:
                refine_result = dict()
                refine_result['frame'] = result['frame']
                if not re.search(r'```json\n(.*)```', result['raw_response'], re.DOTALL):
                    refine_result['result'] = 'no'
                    refine_result['raw_response'] = result['raw_response']
                    refined_results.append(refine_result)
                    continue
                json_str = re.search(r'```json\n(.*)```', result['raw_response'], re.DOTALL).group(1)
                try:
                    json_dict = json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {json_str}")
                    refined_results.append(result)
                    continue
                for key, value in json_dict.items():
                    refine_result[key] = value
                refined_results.append(refine_result)
            else:
                refined_results.append(result)
        save_results(refined_results, self.pair_path)

    def pair_verify(self, skip_frame_nums: int, pairs: list) -> bool:
        """If id has enough medical frames, then return True, otherwise return False.

        Args:
            skip_frame_nums (int): The number of frames to skip.
            frames (list): The processed frames for later use.

        Returns:
            A boolean value that indicates whether the id has enough medical frames.
        """
        if os.path.exists(self.pair_path) and load_results(self.pair_path) != []:
            return True
        for index, frame in tqdm(enumerate(pairs), desc="Pair processing frames"):
            if index % skip_frame_nums != 0:
                continue
            frame_path = os.path.join(self.path, f"keyframe_{frame['frame']}.jpg")
            if os.path.exists(frame_path):
                print(f"Found frame: {frame_path}")
            else:
                print(f"Frame not found: {frame_path}")
            system_prompt = self.pair_system_prompt.format(system=self.system, organ=self.organ, keyword=self.keyword)
            system_prompt = system_prompt.replace("{keyword}", self.keyword).replace("{body_part}", self.organ)
            prompt = self.pair_prompt_template.format(keyword=self.keyword, organ=self.organ, system=self.system, start_time=frame["startTime"], end_time=frame["endTime"], frame_start_time=frame["frameStartTime"], frame_end_time=frame["frameEndTime"], transcript_json_list=json.dumps(frame["sentences"]))
            response = self.openai_interface.run(system_prompt, prompt, image_paths=frame_path)
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                continue
            if response['result'] == 'yes':
                return True
        save_results([], self.pair_path)
        return False