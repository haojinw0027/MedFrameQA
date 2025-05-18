"""Video Process Class."""
import sys
import subprocess
import argparse
import os
import json
import whisper


class VideoProcess:
    """A class for processing video files.

    This class provides methods to handle video-related tasks such as
    search videos from youtube, download videos from youtube, extract key frames and transcribe audio to text.
    """
    def __init__(self, system: str, organ: str, keyword: str, id: str):
        """Initializes a VideoProcess instance.

        Args:
            system (str): The system to process.
            organ (str): The organ to process.
            keyword (str): The keyword to process.
            id (str): The id to process.
        """
        self.skip_download = False
        self.process_status = False
        self.system = system
        self.organ = organ
        self.keyword = keyword
        self.id = id
        self.__parser__()

    def __parser__(self) -> None:
        """Parse the file path."""
        self.path = f'../data/{self.system}/{self.organ}/{self.keyword}/{self.id}'
        video_path = os.path.join(self.path, 'video')
        audio_path = os.path.join(self.path, 'audio')
        if os.path.exists(os.path.join(self.path, 'transcription.json')):
            self.process_status = True
            return
        if os.path.exists(video_path) and os.path.exists(audio_path):
            video_files = os.listdir(video_path)
            audio_files = os.listdir(audio_path)
            if len(video_files) > 0 and len(audio_files) > 0:
                self.skip_download = True
                self.process_status = False
        else:
            self.skip_download = False

    def download_videos_audios(self, video_quality="worstvideo"):
        """Downloads videos and audio files based on the CSV file containing video information.

        Args:
            video_quality (str): The quality of the video to download (e.g., "bestvideo", "worstvideo").

        Returns:
            None
        """
        print("keyword", self.keyword)
        print(f"Downloading video: Keyword={self.keyword}, Video ID={self.id}")
        video_url = f"https://www.youtube.com/watch?v={self.id}"
        output_template_video = os.path.join(self.path, 'video', f"{self.id}_{video_quality}.%(ext)s")
        output_template_audio = os.path.join(self.path, 'audio', f"{self.id}.%(ext)s")
        if not self.is_video_over_a_lower_b(self.id, 5, 120):
            print(f"Video {self.id} is not in the valid duration range. Skipping download.")
            self.skip_download = True
        print(f"skip_download: {self.skip_download}")
        if self.skip_download:
            return
        """Call YouTube API to download video and audio

        ...

        """

    def is_video_over_a_lower_b(self, video_id: str, ta: int = 0, tb: int = 120):
        """Check whether the video duration is over a lower bound and under an upper bound.

        Args:
            video_id (str): The ID of the video to check.
            ta (int): The lower bound in minutes.
            tb (int): The upper bound in minutes.

        Returns:
            bool: True if the video duration is over 30 minutes, False otherwise.
        """
        if tb < ta:
            raise ValueError("The second argument must be greater than the first.")
        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            print(f"Processing video {video_id}")
            """ call youtube api to get the video duration"""
        except subprocess.CalledProcessError as e:
            print(f"Error processing video {video_id}: {e.output.decode().strip()}")
            return False

    def extract_keyframes(self, threshold=0.1, minimum_count=10):
        """Extracts keyframes from all video files in video_dir and saves them to frame_dir.

        Args:
            threshold (float): The threshold for keyframe extraction (range: 0.0 to 1.0).
            minimum_count (int): The minimum number of keyframes to extract.

        Returns:
            None

        Raises:
            subprocess.CalledProcessError: If ffmpeg fails to extract keyframes.
        """
        video_path = os.path.join(self.path, 'video')
        video_files = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
        if not video_files:
            raise FileNotFoundError(f"No video file found in {video_path}")
        if len(video_files) > 1:
            raise RuntimeError(f"Multiple video files found in {video_path}")
        video_file_path = os.path.join(video_path, video_files[0])
        cmd = [
                "ffmpeg",
                "-i", video_file_path,
                "-vf", f"select=gt(scene\\,{threshold}),metadata=print:key=lavfi.scene_score:file={os.path.join(self.path, 'timestamps.txt')}",
                "-vsync", "vfr",
                os.path.join(self.path, f"keyframe_%d.jpg")
        ]
        print(f"Executing keyframe extraction command: {' '.join(cmd)}")
        keyframe_files = [f for f in os.listdir(self.path) if f.startswith("keyframe_") and f.endswith(".jpg")]
        while len(keyframe_files) < minimum_count:
            if threshold < 0.001:
                raise RuntimeError(f"Failed to extract keyframes from {video_file_path}. Only {len(keyframe_files)} keyframes were extracted. Minimum required: {minimum_count}")
            cmd = [
                    "ffmpeg",
                    "-i", video_file_path,
                    "-vf", f"select=gt(scene\\,{threshold}),metadata=print:key=lavfi.scene_score:file={os.path.join(self.path, 'timestamps.txt')}",
                    "-vsync", "vfr",
                    os.path.join(self.path, f"keyframe_%d.jpg")
            ]
            try:
                subprocess.run(cmd, check=True)
                keyframe_files = [f for f in os.listdir(self.path) if f.startswith("keyframe_") and f.endswith(".jpg")]
                if len(keyframe_files) >= minimum_count:
                    print(f"Successfully extracted keyframes from {video_file_path} to {self.path}")
                    break
                else:
                    threshold -= 0.001
                    for file in keyframe_files:
                        os.remove(os.path.join(self.path, file))
                    os.remove(os.path.join(self.path, 'timestamps.txt'))
            except subprocess.CalledProcessError as e:
                print(f"Error extracting keyframes: {e}")
                raise

    def transcribe_audios(self, model_name="large", cuda_count=0):
        """Transcribes audio files in audio_dir using the Whisper model and saves the results as JSON files.

        Args:
            model_name (str, optional): Name of the Whisper model to use. Defaults to "large".
            cuda_count (int, optional): Number of CUDA devices to use. Defaults to 1.

        Returns:
            None
        """
        audio_path = os.path.join(self.path, 'audio')
        audio_files = [f for f in os.listdir(audio_path) if os.path.isfile(os.path.join(audio_path, f))]
        if not audio_files:
            raise FileNotFoundError(f"No audio file found in {audio_path}")
        if len(audio_files) > 1:
            raise RuntimeError(f"Multiple audio files found in {audio_path}")
        output_path = os.path.join(self.path, "transcription.json")
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Transcription file {output_path} already exists and is not empty. Skipping transcription.")
            return
        audio_file_path = os.path.join(audio_path, audio_files[0])
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name, device=f"cuda:{cuda_count}")
        print(f"Transcribing audio file: {audio_file_path}")
        result = model.transcribe(audio_file_path, language="en", verbose=False)
        formatted_results = []
        for segment in result.get("segments", []):
            formatted_results.append({
                "startTime": segment["start"],
                "endTime": segment["end"],
                "sentence": segment["text"].strip()
            })
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(formatted_results, f, ensure_ascii=False, indent=4)
            print(f"Transcription saved to: {output_path}")

    def delete_video_audio(self):
        """Deletes video and audio files."""
        video_path = os.path.join(self.path, 'video')
        audio_path = os.path.join(self.path, 'audio')
        if os.path.exists(video_path):
            for file in os.listdir(video_path):
                os.remove(os.path.join(video_path, file))
            os.rmdir(video_path)
        if os.path.exists(audio_path):
            for file in os.listdir(audio_path):
                os.remove(os.path.join(audio_path, file))
            os.rmdir(audio_path)

    def run(self, mode="search", source="youtube_video", base_output_dir=None, max_downloads_per_keyword=None, threshold=0.1, video_quality="worstvideo"):
        """Executes video processing operations based on the specified mode.

        Depending on the mode, it performs one of the following:
            - 'search': Searches videos and writes the results to a CSV file.
            - 'download': Downloads video and audio files using the CSV file.
            - 'frame': Extracts keyframes from videos.
            - 'transcribe': Transcribes audio files using the Whisper model.

        Args:
            mode (str): The operation mode ('search', 'download', 'frame', or 'transcribe').
            source (str): The source of the videos (e.g., "youtube_video").
            base_output_dir (str): The base directory for storing outputs.
            max_downloads_per_keyword (int, optional): Maximum number of downloads for each keyword.
            threshold (float, optional): Threshold for keyframe extraction. Defaults to 0.3.

        Returns:
            None

        Exits:
            Exits the program if base_output_dir is not provided or mode is unknown.
        """
        if base_output_dir is None:
            print("need base_output_dir parameter")
            sys.exit(1)
        print(f"==== base_output_dir{base_output_dir} ====")
        for category in self.config["categories"]:
            system = category["system"]
            print(f"==== system: {system} ====")
            for body_part in category["body_part"]:
                organ = body_part['organ']
                print(f"==== system: {system} ==== organ: {organ} ====")
                keywords = body_part.get("keywords")
                for keyword in keywords:
                    print(f"==== keyword: {keyword} ====")
                    # Using folder naming for outputs
                    system_folder = system.replace(" ", "_")
                    organ_folder = organ.replace(" ", "_")
                    keyword_folder = keyword.replace(" ", "_")
                    system_organ_keyword_dir = os.path.join(base_output_dir, system_folder, organ_folder, keyword_folder)
                    csv_dir = os.path.join(system_organ_keyword_dir, "ID")
                    video_dir = os.path.join(system_organ_keyword_dir, "videos")
                    audio_dir = os.path.join(system_organ_keyword_dir, "audios")
                    frame_dir = os.path.join(system_organ_keyword_dir, "frames")
                    transcript_dir = os.path.join(system_organ_keyword_dir, "transcripts")
                    os.makedirs(system_organ_keyword_dir, exist_ok=True)
                    os.makedirs(csv_dir, exist_ok=True)
                    os.makedirs(video_dir, exist_ok=True)
                    os.makedirs(audio_dir, exist_ok=True)
                    os.makedirs(frame_dir, exist_ok=True)
                    os.makedirs(transcript_dir, exist_ok=True)
                    if mode == "search":
                        csv_file = os.path.join(csv_dir, f"{system_folder}_{organ_folder}_{keyword_folder}_links.csv")
                        self.search_videos(keyword, csv_file)
                    elif mode == "download":
                        csv_file = os.path.join(csv_dir, f"{system_folder}_{organ_folder}_{keyword_folder}_links.csv")
                        print(f"Processing system: {system}, organ: {organ}, keyword: {keyword}, CSV file: {csv_file}")
                        self.download_videos_audios(csv_file, video_dir, audio_dir, max_downloads_per_keyword, video_quality)
                    elif mode == "frame":
                        self.extract_keyframes(video_dir, frame_dir, threshold)
                    elif mode == "transcribe":
                        self.transcribe_audios(audio_dir, transcript_dir)
                    else:
                        print("Unknown mode. Please choose 'search', 'download', 'extract' or 'transcribe'")
                        sys.exit(1)


def main():
    """Main entry point for the video processing script.

    Parses command line arguments and initiates the video processing operations accordingly.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Retrieve videos, download, extract keyframes, and transcribe audio")
    parser.add_argument("--config", type=str, help="Path to the configuration file (e.g., config.yaml)", default="../config/config.yaml")
    parser.add_argument("--mode", type=str, help="Operation mode: search, download, frame, or transcribe", default="search")
    parser.add_argument("--base_output_dir", type=str, help="Base directory for data outputs")
    parser.add_argument("--source", type=str, help="Video source (e.g., youtube_video)", default="youtube_video")
    parser.add_argument("--max_downloads", type=int, help="Maximum number of videos to download per keyword", default=None)
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for keyframe extraction (range: 0.0 to 1.0)")
    parser.add_argument("--video_quality", type=str, default="worstvideo", help="Video quality for download (e.g., bestvideo, worstvideo)")
    args = parser.parse_args()

    get_video = VideoProcess(args.config)
    get_video.run(mode=args.mode,
                  source=args.source,
                  base_output_dir=args.base_output_dir,
                  max_downloads_per_keyword=args.max_downloads,
                  threshold=args.threshold,
                  video_quality=args.video_quality
                  )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
