"""Process the data."""
import os
import csv
import argparse
from video_process import VideoProcess
from pair_process import PairProcess
from vqa_process import VQAProcess
from utils import get_video_id_from_csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


class Process:
    """Process the data."""
    def __init__(self, system: str, organ: str, keyword: str, id: str, keyframe_threshold: float, keyframe_minimum_count: int, max_frame_num: int, cuda_count: int, skip_frame_nums: int) -> None:
        """Initialize the Process class.

        Args:
            system (str): The system to process.
            organ (str): The organ to process.
            keyword (str): The keyword to process.
            id (str): The id to process.
            keyframe_threshold (float): The threshold to process.
            keyframe_minimum_count (int): The minimum count to process.
            max_frame_num (int): The maximum frame number to generate vqa.
            cuda_count (int): The CUDA count to process.
            skip_frame_nums (int): The number of frames to skip.
        """
        self.system = system
        self.organ = organ
        self.keyword = keyword
        self.id = id
        self.verify_content()
        self.keyframe_threshold = keyframe_threshold
        self.keyframe_minimum_count = keyframe_minimum_count
        self.max_frame_num = max_frame_num
        self.cuda_count = cuda_count
        self.skip_frame_nums = skip_frame_nums

    def verify_content(self, csv_path: str = '../data/30_disease_video_id.csv'):
        """Verify the content."""
        # Check if this combination exists in the CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            exists = False
            for row in reader:
                if row[0] == self.system and row[1] == self.organ and \
                   row[2] == self.keyword and row[3] == self.id:
                    exists = True
                    break
        if not exists:
            raise ValueError(f"No matching entry found in CSV for system={self.system}, organ={self.organ}, keyword={self.keyword}, id={self.id}")

    def download_process(self, video_quality: str) -> None:
        """Download the video.

        Args:
            video_quality (str): The video quality to process.
        """
        video_process = VideoProcess(self.system, self.organ, self.keyword, self.id)
        if video_process.process_status:
            print(f"Video process for {self.system} {self.organ} {self.keyword} {self.id} already exists.")
        elif not video_process.skip_download:
            video_process.download_videos_audios(video_quality=video_quality)
        else:
            print(f"Video process for {self.system} {self.organ} {self.keyword} {self.id} already downloaded, but not processed.")

    def video_process(self, video_quality: str) -> None:
        """Process the video.

        Args:
            video_quality (str): The video quality to process.
        """
        video_process = VideoProcess(self.system, self.organ, self.keyword, self.id)
        if video_process.process_status:
            print(f"Video process for {self.system} {self.organ} {self.keyword} {self.id} already exists.")
            return
        if not video_process.skip_download:
            video_process.download_videos_audios(video_quality=video_quality)
            if video_process.skip_download:
                print(f"Video for {self.system} {self.organ} {self.keyword} {self.id} does not meet duration requirement.")
                return
        if not video_process.process_status:
            video_process.extract_keyframes(threshold=self.keyframe_threshold, minimum_count=self.keyframe_minimum_count)
            video_process.transcribe_audios(cuda_count=self.cuda_count)
            video_process.delete_video_audio()

    def pair_process(self, bias_time: float) -> bool:
        """Process the pair.

        Args:
            bias_time (float): The bias time to process.
        """
        pair_process = PairProcess(self.system, self.organ, self.keyword, self.id)
        if pair_process.status:
            print(f"Pair process for {self.system} {self.organ} {self.keyword} {self.id} already exists.")
            return pair_process.video_invalid
        if pair_process.video_invalid:
            print(f"Video for {self.system} {self.organ} {self.keyword} {self.id} is invalid.")
            return pair_process.video_invalid
        frames = pair_process.parse_timestamps()
        pairs = pair_process.match_text_to_frames(frames, bias_time)
        if not pair_process.pair_verify(self.skip_frame_nums, pairs):
            print(f"Pair process for {self.system} {self.organ} {self.keyword} {self.id} doesn't have enough medical image.")
            return pair_process.video_invalid
        pair_process.pair_process(pairs)
        pair_process.refine_results()
        return pair_process.video_invalid

    def vqa_process(self) -> None:
        """Process the VQA.

        Args:
            max_frame_num (int): The maximum frame number to generate vqa.
        """
        vqa_process = VQAProcess(self.system, self.organ, self.keyword, self.id, self.max_frame_num)
        if vqa_process.vqa_process_status:
            vqa_process.process_pairs()
        else:
            print(f"Pair json is not found for {self.system} {self.organ} {self.keyword} {self.id}, SKIP VQA process.")


def process_csv_row(row: dict, group_keyword: str, keyframe_threshold: float, keyframe_minimum_count: int, video_quality: str, bias_time: float, process_stage: str, max_frame_num: int, cuda_count: int, skip_frame_nums: int) -> None:
    """Process a single row of the CSV file."""
    system = row['system']
    organ = row['organ']
    video_id = row['video_id']
    process = Process(system, organ, group_keyword, video_id, keyframe_threshold, keyframe_minimum_count, max_frame_num, cuda_count, skip_frame_nums)
    if process_stage == "download_process":
        process.download_process(video_quality)
    elif process_stage == "video_process":
        process.video_process(video_quality)
    elif process_stage == "pair_process":
        process.pair_process(bias_time)
        process.vqa_process()


def csv_process(csv_file: str, num_ids: int, keyframe_threshold: float, keyframe_minimum_count: int, video_quality: str, bias_time: float, process_stage: str, max_frame_num: int, system: str, organ: str, keyword: str, cuda_count: int, skip_frame_nums: int) -> None:
    """Process the CSV file.

    Args:
        csv_file (str): The CSV file to process.
        num_ids (int): The number of IDs to process.
        keyframe_threshold (float): The keyframe threshold to process.
        keyframe_minimum_count (int): The keyframe minimum count to process.
        video_quality (str): The video quality to process.
        bias_time (float): The bias time to process.
        process_stage (str): The process stage to process.
        max_frame_num (int): The maximum frame number to generate vqa.
        system (str): The system to process.
        organ (str): The organ to process.
        keyword (str): The keyword to process.
        cuda_count (int): The CUDA count to process.
        skip_frame_nums (int): The number of frames to skip.
    """
    rows = get_video_id_from_csv(csv_file)
    # Group rows by (organ, keyword) if only system is provided
    groups = {}
    if system and not organ and not keyword:
        for row in rows:
            if row['system'] == system:
                k = (row['organ'], row['keyword'])
                groups.setdefault(k, []).append(row)
    elif system and organ and keyword:
        for row in rows:
            if row['system'] == system and row['organ'] == organ and row['keyword'] == keyword:
                k = row['keyword']
                groups.setdefault(k, []).append(row)
    else:
        for row in rows:
            k = row['keyword']
            groups.setdefault(k, []).append(row)

    futures = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for group_key, group in tqdm(groups.items(), desc="Processing groups"):
            group_keyword = group_key[1] if isinstance(group_key, tuple) else group_key
            if num_ids != -1 and process_stage != "pair_process":
                group = group[:min(num_ids, len(group))]
            elif num_ids != -1 and process_stage == "pair_process":
                valid_rows = []
                for row in group:
                    pair_process = PairProcess(row['system'], row['organ'], row['keyword'], row['video_id'])
                    if not pair_process.video_invalid:
                        valid_rows.append(row)
                        if len(valid_rows) >= num_ids:
                            break
                group[:] = valid_rows
                print(f"Filtered rows for {group_keyword}: {len(group)} valid rows found.")
            for row in group:
                futures.append(executor.submit(process_csv_row, row, group_keyword, keyframe_threshold, keyframe_minimum_count, video_quality, bias_time, process_stage, max_frame_num, cuda_count, skip_frame_nums))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
            try:
                future.result()
            except Exception as exc:
                print(f"Row processing generated an exception: {exc}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, default="auxiliary_systems_and_tissues")
    parser.add_argument("--organ", type=str, default="")
    parser.add_argument("--keyword", type=str, default="")
    parser.add_argument("--id", type=str, default="4TmoqtO1Zco")
    parser.add_argument("--video_quality", type=str, default="bestvideo")
    parser.add_argument("--keyframe_threshold", type=float, default=0.1)
    parser.add_argument("--keyframe_minimum_count", type=int, default=10)
    parser.add_argument("--bias_time", type=float, default=20.0)
    parser.add_argument("--process_stage", type=str, choices=["download_process", "video_process", "pair_process", "vqa_process"], default="download_process")
    # csv process
    parser.add_argument("--csv_file", type=str)
    parser.add_argument("--num_ids", type=int, default=-1)  # -1 for all
    parser.add_argument("--max_frame_num", type=int, default=5)  # 999999999 for all
    parser.add_argument("--cuda_count", type=int, default=0)
    parser.add_argument("--skip_frame_nums", type=int, default=5)
    args = parser.parse_args()
    if args.csv_file and args.process_stage == "pair_process":
        csv_process(args.csv_file, args.num_ids, args.keyframe_threshold, args.keyframe_minimum_count, args.video_quality, args.bias_time, args.process_stage, args.max_frame_num, args.system, args.organ, args.keyword, args.cuda_count, args.skip_frame_nums)
        return
    process = Process(args.system, args.organ, args.keyword, args.id, args.keyframe_threshold, args.keyframe_minimum_count, args.max_frame_num, args.cuda_count, args.skip_frame_nums)
    if args.process_stage == "download_process":
        process.download_process(args.video_quality)
    elif args.process_stage == "video_process":
        process.video_process(args.video_quality)
    elif args.process_stage == "pair_process":
        process.pair_process(args.bias_time)
    elif args.process_stage == "vqa_process":
        process.vqa_process()


if __name__ == "__main__":
    main()
