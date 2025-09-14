import streamlit as st
import re
from yt_dlp import YoutubeDL
import os
import subprocess
import json
import tempfile
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import zipfile
from pathlib import Path
import threading
import queue
import time
from dataclasses import dataclass, asdict
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Enhanced Configuration
CONFIG = {
    "DOWNLOADS_DIR": "downloads",
    "BATCH_DIR": "batch_downloads",
    "LOGS_DIR": "logs",
    "CACHE_DIR": "cache",
    "TEMP_DIR": "temp",
    "MAX_WORKERS": 3,
    "CACHE_EXPIRY_HOURS": 24,
    "MAX_FILE_SIZE_MB": 500,
    "SUPPORTED_FORMATS": ["mp4", "mkv", "webm", "avi", "mov"],
    "QUALITY_PRESETS": {
        "Ultra HD": "2160p",
        "Full HD": "1080p", 
        "HD": "720p",
        "SD": "480p",
        "Mobile": "360p"
    }
}

@dataclass
class VideoInfo:
    video_id: str
    title: str
    description: str
    thumbnail: str
    duration: str
    view_count: int
    like_count: int
    uploader: str
    upload_date: str
    formats: List[Dict]
    tags: List[str]
    category: str
    age_limit: int
    is_live: bool
    webpage_url: str

@dataclass
class ProcessingJob:
    id: str
    url: str
    status: str
    progress: float
    output_path: Optional[str]
    error: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

class SystemChecker:
    @staticmethod
    def check_ffmpeg() -> Tuple[bool, str]:
        """Enhanced FFmpeg detection with version info."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                return True, version_line
            return False, "FFmpeg found but not working properly"
        except subprocess.TimeoutExpired:
            return False, "FFmpeg timeout - may be corrupted"
        except FileNotFoundError:
            return False, "FFmpeg not found in PATH"
        except Exception as e:
            return False, f"FFmpeg check failed: {str(e)}"
    
    @staticmethod
    def check_system_resources() -> Dict:
        """Check system resources and capabilities."""
        import psutil
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_free_gb": psutil.disk_usage('/').free / (1024**3),
                "cpu_count": psutil.cpu_count()
            }
        except ImportError:
            return {"error": "psutil not installed - install for system monitoring"}

class CacheManager:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def get_cached_info(self, url: str) -> Optional[Dict]:
        """Retrieve cached video info if not expired."""
        cache_key = self.get_cache_key(url)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                    
                cache_time = datetime.fromisoformat(cached['cached_at'])
                if datetime.now() - cache_time < timedelta(hours=CONFIG['CACHE_EXPIRY_HOURS']):
                    return cached['data']
            except Exception:
                pass
        return None
    
    def cache_info(self, url: str, info: Dict):
        """Cache video information."""
        cache_key = self.get_cache_key(url)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            cached_data = {
                'cached_at': datetime.now().isoformat(),
                'data': info
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.warning(f"Failed to cache data: {e}")

class AdvancedYouTubeProcessor:
    def __init__(self):
        self.create_directories()
        self.ffmpeg_available, self.ffmpeg_info = SystemChecker.check_ffmpeg()
        self.cache_manager = CacheManager(CONFIG['CACHE_DIR'])
        self.processing_jobs: Dict[str, ProcessingJob] = {}
        
        if not self.ffmpeg_available:
            st.error(f"⚠️ FFmpeg Issue: {self.ffmpeg_info}")
            st.info("💡 Install FFmpeg: https://ffmpeg.org/download.html")
        else:
            st.success(f"✅ {self.ffmpeg_info}")
    
    def create_directories(self):
        """Create all necessary directories."""
        for directory in CONFIG.values():
            if isinstance(directory, str) and directory.endswith('_DIR'):
                Path(directory).mkdir(exist_ok=True)
    
    def extract_youtube_id(self, url: str) -> Optional[str]:
        """Enhanced YouTube ID extraction with validation."""
        patterns = [
            r"(?:v=|youtu\.be\/|shorts\/)([a-zA-Z0-9_-]{11})",
            r"youtube\.com\/embed\/([a-zA-Z0-9_-]{11})",
            r"youtube\.com\/v\/([a-zA-Z0-9_-]{11})",
            r"youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                if len(video_id) == 11:  # YouTube video IDs are always 11 characters
                    return video_id
        return None

    def validate_video(self, url: str, use_cache: bool = True) -> Dict:
        """Enhanced video validation with comprehensive info extraction."""
        if use_cache:
            cached_info = self.cache_manager.get_cached_info(url)
            if cached_info:
                return cached_info
        
        try:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
                "writeinfojson": False,
                "writedescription": False,
                "writesubtitles": False
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            
            if not info or not info.get("id"):
                return {"error": "Video tidak tersedia atau dibatasi"}
            
            # Enhanced information extraction
            duration = info.get("duration", 0) or 0
            duration_str = str(timedelta(seconds=int(duration))) if duration else "Live/Unknown"
            
            # Extract comprehensive format information
            formats = self.extract_enhanced_formats(info)
            
            # Build comprehensive video info
            video_info = {
                "video_id": info.get("id", ""),
                "title": info.get("title", "Unknown Title"),
                "description": (info.get("description", "") or "")[:1000],
                "thumbnail": info.get("thumbnail", ""),
                "duration": duration_str,
                "duration_seconds": duration,
                "view_count": info.get("view_count", 0) or 0,
                "like_count": info.get("like_count", 0) or 0,
                "dislike_count": info.get("dislike_count", 0) or 0,
                "uploader": info.get("uploader", "Unknown"),
                "upload_date": info.get("upload_date", ""),
                "formats": formats,
                "tags": info.get("tags", []) or [],
                "category": info.get("category", "Unknown"),
                "age_limit": info.get("age_limit", 0) or 0,
                "is_live": info.get("is_live", False),
                "webpage_url": info.get("webpage_url", url),
                "channel_id": info.get("channel_id", ""),
                "subscriber_count": info.get("channel_follower_count", 0) or 0,
                "fps": info.get("fps", 0) or 0,
                "resolution": f"{info.get('width', 0)}x{info.get('height', 0)}",
                "filesize_approx": info.get("filesize_approx", 0) or 0
            }
            
            if use_cache:
                self.cache_manager.cache_info(url, video_info)
            
            return video_info
            
        except Exception as e:
            error_msg = str(e)
            return {
                "error": f"Gagal mengakses video: {error_msg}. "
                        f"Kemungkinan: video private, age-restricted, geo-blocked, atau yt-dlp perlu update."
            }

    def extract_enhanced_formats(self, info: dict) -> List[Dict]:
        """Extract comprehensive format information with quality analysis."""
        formats = []
        video_formats = [f for f in info.get("formats", []) if f.get("vcodec") != "none"]
        
        for fmt in video_formats:
            filesize = fmt.get("filesize") or fmt.get("filesize_approx") or 0
            
            # Enhanced format information
            format_info = {
                "format_id": fmt.get("format_id", ""),
                "ext": fmt.get("ext", "mp4"),
                "quality": fmt.get("format_note", "") or fmt.get("quality", "unknown"),
                "filesize": filesize,
                "filesize_mb": round(filesize / (1024*1024), 1) if filesize else 0,
                "fps": fmt.get("fps", 0) or 0,
                "width": fmt.get("width", 0) or 0,
                "height": fmt.get("height", 0) or 0,
                "resolution": f"{fmt.get('width', 0) or 0}x{fmt.get('height', 0) or 0}",
                "vcodec": fmt.get("vcodec", "unknown"),
                "acodec": fmt.get("acodec", "unknown"),
                "abr": fmt.get("abr", 0) or 0,
                "vbr": fmt.get("vbr", 0) or 0,
                "protocol": fmt.get("protocol", "https"),
                "quality_rank": self.calculate_quality_rank(fmt)
            }
            formats.append(format_info)
        
        # Sort by quality rank (higher is better)
        return sorted(formats, key=lambda x: x["quality_rank"], reverse=True)

    def calculate_quality_rank(self, fmt: dict) -> int:
        """Calculate quality ranking for format sorting."""
        height = fmt.get("height", 0) or 0
        fps = fmt.get("fps", 0) or 0
        filesize = fmt.get("filesize") or fmt.get("filesize_approx") or 0
        
        # Base score from resolution
        score = height * 10
        
        # Bonus for higher FPS
        if fps >= 60:
            score += 1000
        elif fps >= 30:
            score += 500
            
        # Bonus for reasonable filesize (not too small, not too large)
        if 10*1024*1024 <= filesize <= 500*1024*1024:  # 10MB to 500MB
            score += 100
        
        return score

    def generate_advanced_embed_code(
        self, 
        video_id: str, 
        aspect_ratio: str = "16:9", 
        autoplay: bool = False,
        start_time: int = 0, 
        end_time: int = 0,
        controls: bool = True,
        modestbranding: bool = False,
        loop: bool = False,
        privacy_enhanced: bool = True
    ) -> Dict[str, str]:
        """Generate multiple embed code variants with advanced options."""
        
        domain = "youtube-nocookie.com" if privacy_enhanced else "youtube.com"
        
        # Aspect ratio calculations
        aspect_configs = {
            "16:9": {"width": "560", "height": "315", "padding": "56.25%"},
            "9:16": {"width": "315", "height": "560", "padding": "177.78%"},
            "4:3": {"width": "560", "height": "420", "padding": "75%"},
            "1:1": {"width": "400", "height": "400", "padding": "100%"},
            "21:9": {"width": "560", "height": "240", "padding": "42.86%"}
        }
        
        config = aspect_configs.get(aspect_ratio, aspect_configs["16:9"])
        
        # Build parameters
        params = []
        if not controls:
            params.append("controls=0")
        if modestbranding:
            params.append("modestbranding=1")
        if autoplay:
            params.append("autoplay=1&mute=1")
        if loop:
            params.append("loop=1&playlist=" + video_id)
        if start_time:
            params.append(f"start={start_time}")
        if end_time:
            params.append(f"end={end_time}")
        
        params.append("rel=0")
        param_string = "&".join(params)
        
        # Generate different embed variants
        embeds = {
            "responsive": f"""
<div style="position: relative; width: 100%; max-width: {config['width']}px; margin: auto; height: 0; padding-bottom: {config['padding']};">
    <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
        src="https://www.{domain}/embed/{video_id}?{param_string}"
        frameborder="0" allowfullscreen
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture">
    </iframe>
</div>""",
            
            "fixed": f"""
<iframe width="{config['width']}" height="{config['height']}"
    src="https://www.{domain}/embed/{video_id}?{param_string}"
    frameborder="0" allowfullscreen
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture">
</iframe>""",
            
            "amp": f"""
<amp-youtube data-videoid="{video_id}" layout="responsive"
    width="{config['width']}" height="{config['height']}">
</amp-youtube>""",
            
            "oembed_url": f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        }
        
        return embeds

    def advanced_video_processing(
        self, 
        input_path: str, 
        output_path: str, 
        options: Dict
    ) -> bool:
        """Advanced video processing with multiple options."""
        try:
            cmd = ["ffmpeg", "-i", input_path]
            
            # Video filters
            filters = []
            
            # Cropping
            if options.get("crop_aspect"):
                aspect_filters = {
                    "9:16": "crop=ih*9/16:ih",
                    "16:9": "crop=iw:iw*9/16", 
                    "4:3": "crop=iw:iw*3/4",
                    "1:1": "crop=min(iw\\,ih):min(iw\\,ih)"
                }
                if options["crop_aspect"] in aspect_filters:
                    filters.append(aspect_filters[options["crop_aspect"]])
            
            # Scaling
            if options.get("scale_height"):
                filters.append(f"scale=-2:{options['scale_height']}")
            
            # Frame rate
            if options.get("fps"):
                filters.append(f"fps={options['fps']}")
            
            # Add filters to command
            if filters:
                cmd.extend(["-vf", ",".join(filters)])
            
            # Video codec options
            if options.get("video_codec", "h264") == "h265":
                cmd.extend(["-c:v", "libx265", "-preset", "medium"])
            else:
                cmd.extend(["-c:v", "libx264", "-preset", "fast"])
            
            # Audio options
            if options.get("audio_codec"):
                cmd.extend(["-c:a", options["audio_codec"]])
            else:
                cmd.extend(["-c:a", "aac"])
            
            # Quality options
            if options.get("crf"):
                cmd.extend(["-crf", str(options["crf"])])
            
            # Output format
            cmd.extend(["-f", "mp4", "-y", output_path])
            
            # Execute with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                check=True
            )
            
            return os.path.exists(output_path) and os.path.getsize(output_path) > 1024
            
        except subprocess.TimeoutExpired:
            st.error("⏱️ Processing timeout - file too large or complex")
            return False
        except subprocess.CalledProcessError as e:
            st.error(f"🔧 Processing failed: {e.stderr}")
            return False
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return False

    def download_with_progress(
        self, 
        url: str, 
        output_path: str,
        format_id: str = "best",
        processing_options: Optional[Dict] = None
    ) -> Optional[str]:
        """Download with real-time progress tracking."""
        
        def progress_hook(d):
            if d['status'] == 'downloading':
                if 'total_bytes' in d:
                    progress = d['downloaded_bytes'] / d['total_bytes']
                    st.session_state.download_progress = progress
            elif d['status'] == 'finished':
                st.session_state.download_progress = 1.0
        
        try:
            video_id = self.extract_youtube_id(url)
            if not video_id:
                return None
            
            # Setup download options
            ydl_opts = {
                "format": format_id,
                "outtmpl": os.path.join(output_path, f"video-{video_id}.%(ext)s"),
                "merge_output_format": "mp4",
                "progress_hooks": [progress_hook],
                "quiet": True,
                "no_warnings": True
            }
            
            # Download
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_file = os.path.join(output_path, f"video-{info['id']}.mp4")
            
            if not os.path.exists(downloaded_file):
                return None
            
            # Apply additional processing if specified
            if processing_options and self.ffmpeg_available:
                processed_file = os.path.join(output_path, f"video-{video_id}-processed.mp4")
                if self.advanced_video_processing(downloaded_file, processed_file, processing_options):
                    os.remove(downloaded_file)
                    return processed_file
            
            return downloaded_file
            
        except Exception as e:
            st.error(f"Download failed: {e}")
            return None

    def batch_process_with_threading(
        self, 
        urls: List[str], 
        processing_options: Dict
    ) -> List[str]:
        """Process multiple videos with threading for better performance."""
        
        results = []
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        def process_single_video(url: str, index: int) -> Optional[str]:
            try:
                status_container.text(f"Processing {index+1}/{len(urls)}: {url[:50]}...")
                
                output_dir = os.path.join(CONFIG['BATCH_DIR'], f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(output_dir, exist_ok=True)
                
                result = self.download_with_progress(url, output_dir, processing_options=processing_options)
                progress_bar.progress((index + 1) / len(urls))
                
                return result
            except Exception as e:
                st.error(f"Failed to process {url}: {e}")
                return None
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(CONFIG['MAX_WORKERS'], len(urls))) as executor:
            future_to_url = {
                executor.submit(process_single_video, url, i): (url, i) 
                for i, url in enumerate(urls)
            }
            
            for future in as_completed(future_to_url):
                url, index = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    st.error(f"Thread error for {url}: {e}")
        
        return results

def create_advanced_ui():
    """Create the advanced user interface."""
    st.set_page_config(
        page_title="YouTube Professional Suite",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Advanced CSS styling
    st.markdown("""
        <style>
        :root {
            --primary-color: #ff0000;
            --secondary-color: #cc0000;
            --accent-color: #1976d2;
            --success-color: #4caf50;
            --warning-color: #ff9800;
            --error-color: #f44336;
        }
        
        .main-header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(255,0,0,0.3);
        }
        
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .feature-card {
    background: #000000; /* putih polos */
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid var(--accent-color); /* tetap ada garis aksen */
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

        
        .video-info-card {
            background: linear-gradient(145deg, #e3f2fd, #bbdefb);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border-left: 4px solid var(--accent-color);
        }
        
        .processing-status {
            background: linear-gradient(145deg, #fff3e0, #ffcc02);
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 16px rgba(255,0,0,0.3);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, var(--secondary-color), #aa0000);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255,0,0,0.4);
        }
        
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online { background-color: var(--success-color); }
        .status-warning { background-color: var(--warning-color); }
        .status-error { background-color: var(--error-color); }
        </style>
    """, unsafe_allow_html=True)

def main():
    create_advanced_ui()
    
    processor = AdvancedYouTubeProcessor()
    
    # Initialize session state
    if 'download_progress' not in st.session_state:
        st.session_state.download_progress = 0.0
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🎬 YouTube Professional Suite</h1>
            <p>Advanced video processing, batch operations, and professional tools</p>
            <div style="font-size: 0.9em; margin-top: 1rem;">
                <span class="status-indicator status-online"></span>Online • 
                <span class="status-indicator status-online"></span>yt-dlp Ready •
                <span class="status-indicator {}"></span>FFmpeg {}
            </div>
        </div>
    """.format(
        "status-online" if processor.ffmpeg_available else "status-error",
        "Ready" if processor.ffmpeg_available else "Not Available"
    ), unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("🔧 Advanced Settings")
        
        # Operation Mode
        operation_mode = st.selectbox(
            "🎯 Operation Mode:",
            [
                "🎥 Single Video Pro",
                "📦 Batch Processing", 
                "📋 Playlist Manager",
                "🎵 Audio Extraction",
                "🔧 Advanced Processing",
                "📊 Analytics Mode"
            ]
        )
        
        st.divider()
        
        # Processing Options
        with st.expander("🎨 Video Processing", expanded=True):
            aspect_ratio = st.selectbox(
                "Aspect Ratio:",
                ["16:9 (YouTube)", "9:16 (Shorts/TikTok)", "4:3 (Classic)", "1:1 (Instagram)", "21:9 (Cinematic)"]
            )
            
            quality_preset = st.selectbox(
                "Quality Preset:",
                list(CONFIG['QUALITY_PRESETS'].keys())
            )
            
            video_codec = st.selectbox(
                "Video Codec:",
                ["h264 (Compatible)", "h265 (Efficient)"]
            )
            
            enable_cropping = st.checkbox("🔲 Auto-crop to aspect ratio", value=True)
            enable_optimization = st.checkbox("⚡ Optimize for web", value=True)
        
        # Embed Options
        with st.expander("🎬 Embed Settings"):
            privacy_enhanced = st.checkbox("🔐 Privacy Enhanced Mode", value=True)
            autoplay = st.checkbox("▶️ Autoplay (muted)")
            show_controls = st.checkbox("🎛️ Show controls", value=True)
            modestbranding = st.checkbox("🏷️ Modest branding")
            loop_video = st.checkbox("🔄 Loop video")
            start_time = st.number_input("⏰ Start time (seconds)", min_value=0, value=0)
            end_time = st.number_input("⏱️ End time (seconds)", min_value=0, value=0)
        
        # System Status
        st.divider()
        system_info = SystemChecker.check_system_resources()
        if "error" not in system_info:
            st.metric("💾 Memory Usage", f"{system_info.get('memory_percent', 0):.1f}%")
            st.metric("💿 Disk Free", f"{system_info.get('disk_free_gb', 0):.1f} GB")
    
    # Main Content Area
    main_col, info_col = st.columns([2.5, 1])
    
    with main_col:
        if "Single Video Pro" in operation_mode:
            st.subheader("🎥 Professional Single Video Processing")
            
            # URL Input with validation
            url = st.text_input(
                "🔗 YouTube URL:",
                placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
                help="Supports all YouTube URL formats including Shorts"
            )
            
            # Real-time URL validation
            if url:
                video_id = processor.extract_youtube_id(url)
                if video_id:
                    st.success(f"✅ Valid YouTube URL detected (ID: {video_id})")
                    
                    # Cache toggle
                    use_cache = st.checkbox("⚡ Use cached data", value=True, 
                                          help="Faster loading for recently analyzed videos")
                    
                    # Analyze video
                    if st.button("🔍 Analyze Video", use_container_width=True):
                        with st.spinner("🔄 Analyzing video metadata..."):
                            video_info = processor.validate_video(url, use_cache=use_cache)
                        
                        if video_info.get("error"):
                            st.error(f"❌ {video_info['error']}")
                        else:
                            # Store video info in session state
                            st.session_state.current_video_info = video_info
                            st.success("✅ Video analyzed successfully!")
                    
                    # Display video information if available
                    if hasattr(st.session_state, 'current_video_info'):
                        video_info = st.session_state.current_video_info
                        
                        # Video info card
                        st.markdown('<div class="video-info-card">', unsafe_allow_html=True)
                        
                        col_thumb, col_details = st.columns([1, 2])
                        
                        with col_thumb:
                            if video_info.get("thumbnail"):
                                st.image(video_info["thumbnail"], use_container_width=True)
                        
                        with col_details:
                            st.markdown(f"**📺 {video_info.get('title', 'Unknown')}**")
                            st.write(f"👤 **Channel:** {video_info.get('uploader', 'Unknown')}")
                            st.write(f"⏱️ **Duration:** {video_info.get('duration', 'Unknown')}")
                            st.write(f"👁️ **Views:** {video_info.get('view_count', 0):,}")
                            st.write(f"👍 **Likes:** {video_info.get('like_count', 0):,}")
                            st.write(f"📅 **Upload:** {video_info.get('upload_date', 'Unknown')}")
                            
                            if video_info.get('is_live'):
                                st.warning("🔴 Live Stream")
                            
                            if video_info.get('age_limit', 0) > 0:
                                st.warning(f"🔞 Age Restricted: {video_info['age_limit']}+")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Format selection
                        if video_info.get("formats"):
                            st.subheader("📊 Available Formats")
                            
                            # Create format selection table
                            format_data = []
                            for fmt in video_info["formats"][:15]:  # Show top 15 formats
                                format_data.append({
                                    "Quality": fmt.get("quality", "unknown"),
                                    "Resolution": fmt.get("resolution", "0x0"),
                                    "FPS": fmt.get("fps", 0),
                                    "Size (MB)": fmt.get("filesize_mb", 0),
                                    "Codec": fmt.get("vcodec", "unknown"),
                                    "Format ID": fmt.get("format_id", "")
                                })
                            
                            format_df = st.dataframe(format_data, use_container_width=True)
                            
                            # Format selection dropdown
                            format_options = {
                                f"{fmt['quality']} ({fmt['resolution']}) - {fmt['filesize_mb']}MB": fmt['format_id']
                                for fmt in video_info["formats"][:10]
                                if fmt.get('format_id')
                            }
                            format_options["🎯 Best Quality (Auto)"] = "best"
                            format_options["💾 Best Size/Quality Balance"] = "best[height<=720]"
                            
                            selected_format_display = st.selectbox(
                                "🎯 Select Download Format:", 
                                list(format_options.keys())
                            )
                            selected_format = format_options[selected_format_display]
                        else:
                            selected_format = "best"
                        
                        # Embed code generation
                        st.subheader("🎬 Embed Code Generator")
                        
                        aspect = aspect_ratio.split(" ")[0]
                        embed_codes = processor.generate_advanced_embed_code(
                            video_info["video_id"],
                            aspect,
                            autoplay,
                            start_time,
                            end_time,
                            show_controls,
                            modestbranding,
                            loop_video,
                            privacy_enhanced
                        )
                        
                        # Embed preview
                        st.markdown("**Preview:**")
                        st.markdown(embed_codes["responsive"], unsafe_allow_html=True)
                        
                        # Embed code variants
                        embed_tab1, embed_tab2, embed_tab3, embed_tab4 = st.tabs([
                            "📱 Responsive", "🖥️ Fixed Size", "⚡ AMP", "🔗 oEmbed"
                        ])
                        
                        with embed_tab1:
                            st.code(embed_codes["responsive"], language="html")
                        
                        with embed_tab2:
                            st.code(embed_codes["fixed"], language="html")
                        
                        with embed_tab3:
                            st.code(embed_codes["amp"], language="html")
                        
                        with embed_tab4:
                            st.write("**oEmbed API URL:**")
                            st.code(embed_codes["oembed_url"])
                        
                        # Action buttons
                        st.subheader("🚀 Actions")
                        
                        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
                        
                        with col_btn1:
                            if st.button("⬇️ Download Video", use_container_width=True):
                                processing_options = {
                                    "crop_aspect": aspect if enable_cropping else None,
                                    "video_codec": video_codec.split(" ")[0],
                                    "scale_height": CONFIG['QUALITY_PRESETS'].get(quality_preset, "720p").replace("p", "") if enable_optimization else None,
                                    "crf": 23 if enable_optimization else None
                                }
                                
                                with st.spinner("📥 Downloading and processing..."):
                                    # Show progress bar
                                    progress_placeholder = st.empty()
                                    
                                    output_dir = CONFIG['DOWNLOADS_DIR']
                                    file_path = processor.download_with_progress(
                                        url, output_dir, selected_format, processing_options
                                    )
                                    
                                    if file_path and os.path.exists(file_path):
                                        st.success("✅ Download completed!")
                                        
                                        with open(file_path, "rb") as file:
                                            st.download_button(
                                                label="💾 Download MP4 File",
                                                data=file,
                                                file_name=f"{video_info['title'][:50]}.mp4",
                                                mime="video/mp4",
                                                use_container_width=True
                                            )
                                        
                                        # Cleanup
                                        try:
                                            os.remove(file_path)
                                        except:
                                            pass
                                    else:
                                        st.error("❌ Download failed!")
                        
                        with col_btn2:
                            if st.button("🎵 Extract Audio", use_container_width=True):
                                with st.spinner("🎵 Extracting audio..."):
                                    # Audio extraction logic here
                                    ydl_opts = {
                                        "format": "bestaudio[ext=m4a]/bestaudio",
                                        "outtmpl": os.path.join(CONFIG['DOWNLOADS_DIR'], f"audio-{video_info['video_id']}.%(ext)s"),
                                        "quiet": True
                                    }
                                    
                                    try:
                                        with YoutubeDL(ydl_opts) as ydl:
                                            ydl.download([url])
                                        
                                        audio_file = os.path.join(CONFIG['DOWNLOADS_DIR'], f"audio-{video_info['video_id']}.m4a")
                                        if os.path.exists(audio_file):
                                            with open(audio_file, "rb") as file:
                                                st.download_button(
                                                    label="💾 Download Audio",
                                                    data=file,
                                                    file_name=f"{video_info['title'][:50]}.m4a",
                                                    mime="audio/mp4",
                                                    use_container_width=True
                                                )
                                            os.remove(audio_file)
                                        else:
                                            st.error("❌ Audio extraction failed!")
                                    except Exception as e:
                                        st.error(f"❌ Audio extraction error: {e}")
                        
                        with col_btn3:
                            if st.button("📋 Copy Embed", use_container_width=True):
                                st.code(embed_codes["responsive"], language="html")
                                st.success("✅ Responsive embed code displayed above!")
                        
                        with col_btn4:
                            if st.button("📊 Analyze More", use_container_width=True):
                                # Advanced analytics
                                st.subheader("📊 Video Analytics")
                                
                                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                                
                                with col_metric1:
                                    st.metric(
                                        "Engagement Rate",
                                        f"{(video_info.get('like_count', 0) / max(video_info.get('view_count', 1), 1) * 100):.2f}%"
                                    )
                                
                                with col_metric2:
                                    st.metric(
                                        "Duration",
                                        video_info.get('duration', 'Unknown')
                                    )
                                
                                with col_metric3:
                                    st.metric(
                                        "Resolution",
                                        video_info.get('resolution', 'Unknown')
                                    )
                                
                                with col_metric4:
                                    st.metric(
                                        "File Size (Est.)",
                                        f"{video_info.get('filesize_approx', 0) / (1024*1024):.1f} MB" if video_info.get('filesize_approx') else "Unknown"
                                    )
                                
                                # Tags and category
                                if video_info.get('tags'):
                                    st.write("**🏷️ Tags:**")
                                    st.write(", ".join(video_info['tags'][:20]))  # Show first 20 tags
                                
                                if video_info.get('category'):
                                    st.write(f"**📂 Category:** {video_info['category']}")
                else:
                    st.warning("⚠️ Invalid YouTube URL format")
        
        elif "Batch Processing" in operation_mode:
            st.subheader("📦 Advanced Batch Processing")
            
            st.info("💡 Process multiple videos simultaneously with advanced options")
            
            # Batch input methods
            input_method = st.radio(
                "📥 Input Method:",
                ["📝 Manual URLs", "📄 Upload File", "🔗 From Playlist"]
            )
            
            urls = []
            
            if input_method == "📝 Manual URLs":
                urls_input = st.text_area(
                    "🔗 YouTube URLs (one per line):",
                    placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=...\nhttps://youtu.be/...",
                    height=200
                )
                urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            
            elif input_method == "📄 Upload File":
                uploaded_file = st.file_uploader(
                    "Upload text file with URLs", 
                    type=['txt', 'csv'],
                    help="Upload a .txt or .csv file with YouTube URLs"
                )
                
                if uploaded_file:
                    try:
                        content = uploaded_file.read().decode('utf-8')
                        urls = [url.strip() for url in content.split('\n') if url.strip() and 'youtube.com' in url or 'youtu.be' in url]
                        st.success(f"✅ Loaded {len(urls)} URLs from file")
                    except Exception as e:
                        st.error(f"❌ Error reading file: {e}")
            
            elif input_method == "🔗 From Playlist":
                playlist_url = st.text_input(
                    "🔗 YouTube Playlist URL:",
                    placeholder="https://www.youtube.com/playlist?list=..."
                )
                
                if playlist_url and st.button("🔍 Extract URLs from Playlist"):
                    with st.spinner("📋 Extracting playlist..."):
                        urls = processor.extract_playlist_urls(playlist_url)
                        if urls:
                            st.success(f"✅ Extracted {len(urls)} URLs from playlist")
                            st.text_area("Extracted URLs:", "\n".join(urls), height=150)
                        else:
                            st.error("❌ Failed to extract playlist URLs")
            
            # Batch processing options
            if urls:
                st.subheader("⚙️ Batch Processing Settings")
                
                col_settings1, col_settings2 = st.columns(2)
                
                with col_settings1:
                    batch_quality = st.selectbox(
                        "📊 Quality for all videos:",
                        ["best", "720p", "480p", "360p"]
                    )
                    
                    max_concurrent = st.slider(
                        "🔀 Concurrent downloads:",
                        min_value=1,
                        max_value=5,
                        value=2,
                        help="More concurrent downloads = faster but uses more resources"
                    )
                
                with col_settings2:
                    batch_format = st.selectbox(
                        "📁 Output format:",
                        ["mp4", "mkv", "webm"]
                    )
                    
                    include_metadata = st.checkbox(
                        "📋 Include metadata files",
                        help="Save video information as JSON files"
                    )
                
                # Processing preview
                st.info(f"📊 Ready to process {len(urls)} videos with {max_concurrent} concurrent downloads")
                
                if st.button("🚀 Start Batch Processing", use_container_width=True):
                    processing_options = {
                        "crop_aspect": aspect_ratio.split(" ")[0] if enable_cropping else None,
                        "video_codec": video_codec.split(" ")[0],
                        "quality": batch_quality
                    }
                    
                    with st.spinner("🔄 Processing batch..."):
                        progress_container = st.container()
                        
                        with progress_container:
                            st.markdown('<div class="processing-status">', unsafe_allow_html=True)
                            st.write("📊 **Batch Processing Status**")
                            
                            overall_progress = st.progress(0)
                            status_text = st.empty()
                            
                            # Start batch processing
                            results = processor.batch_process_with_threading(urls, processing_options)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        if results:
                            # Create ZIP file
                            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                            zip_path = os.path.join(CONFIG['BATCH_DIR'], f"batch_{batch_id}.zip")
                            
                            with zipfile.ZipFile(zip_path, 'w') as zipf:
                                for file_path in results:
                                    if os.path.exists(file_path):
                                        zipf.write(file_path, os.path.basename(file_path))
                            
                            # Offer download
                            if os.path.exists(zip_path):
                                with open(zip_path, "rb") as file:
                                    st.download_button(
                                        label=f"💾 Download Batch Archive ({len(results)} files)",
                                        data=file,
                                        file_name=f"batch_download_{batch_id}.zip",
                                        mime="application/zip",
                                        use_container_width=True
                                    )
                                
                                # Cleanup
                                for file_path in results:
                                    try:
                                        os.remove(file_path)
                                    except:
                                        pass
                                
                                try:
                                    os.remove(zip_path)
                                except:
                                    pass
                            
                            st.success(f"✅ Batch processing completed! {len(results)} videos processed successfully.")
                        else:
                            st.error("❌ Batch processing failed - no videos were processed successfully")
    
    # Information sidebar
    with info_col:
        st.markdown("""
            <div class="feature-card">
                <h3>🚀 Pro Features</h3>
                <ul>
                    <li>✅ Advanced video processing</li>
                    <li>✅ Multiple aspect ratios</li>
                    <li>✅ Batch operations</li>
                    <li>✅ Smart caching</li>
                    <li>✅ Real-time progress</li>
                    <li>✅ Format optimization</li>
                    <li>✅ Privacy-enhanced embeds</li>
                    <li>✅ System monitoring</li>
                    <li>✅ Threaded processing</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h3>📊 System Status</h3>
                <div class="metric-card">
                    <div><span class="status-indicator status-online"></span>Service Online</div>
                    <div><span class="status-indicator {}"></span>FFmpeg {}</div>
                    <div><span class="status-indicator status-online"></span>Cache Active</div>
                </div>
            </div>
        """.format(
            "status-online" if processor.ffmpeg_available else "status-error",
            "Ready" if processor.ffmpeg_available else "Missing"
        ), unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h3>💡 Tips & Tricks</h3>
                <ul>
                    <li>🎯 Use "best" quality for automatic selection</li>
                    <li>⚡ Enable caching for faster repeated analysis</li>
                    <li>🔲 Auto-crop converts videos to target aspect ratio</li>
                    <li>🎵 Audio extraction preserves highest quality</li>
                    <li>📦 Batch processing is optimized for multiple videos</li>
                    <li>🔐 Privacy mode uses youtube-nocookie.com</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <h4>🎬 YouTube Professional Suite</h4>
            <p>Built with ❤️ using Streamlit, yt-dlp, and FFmpeg</p>
            <p><small>⚖️ Legal Notice: Only download content you have permission to use. Respect copyright laws and content creators' rights.</small></p>
            <p><small>🔧 For best results, ensure FFmpeg is installed and updated. Keep yt-dlp current with: <code>pip install --upgrade yt-dlp</code></small></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()