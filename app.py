import streamlit as st
import re
from yt_dlp import YoutubeDL
import os
import subprocess
import json
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

# Configuration
CONFIG = {
    "DOWNLOADS_DIR": "downloads",
    "CACHE_DIR": "cache",
    "LOGS_DIR": "logs",
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

class SystemChecker:
    @staticmethod
    def check_ffmpeg() -> tuple[bool, str]:
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
        try:
            import psutil
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
                if datetime.now() - cache_time < timedelta(hours=24):
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
        
        if not self.ffmpeg_available:
            st.error(f"⚠️ FFmpeg Issue: {self.ffmpeg_info}")
            st.info("💡 Install FFmpeg: https://ffmpeg.org/download.html")
        else:
            st.success(f"✅ {self.ffmpeg_info}")
    
    def create_directories(self):
        """Create necessary directories."""
        for directory in [CONFIG['DOWNLOADS_DIR'], CONFIG['CACHE_DIR'], CONFIG['LOGS_DIR']]:
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
                if len(video_id) == 11:
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
            
            duration = info.get("duration", 0) or 0
            duration_str = str(timedelta(seconds=int(duration))) if duration else "Live/Unknown"
            
            formats = self.extract_enhanced_formats(info)
            
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
        
        return sorted(formats, key=lambda x: x["quality_rank"], reverse=True)

    def calculate_quality_rank(self, fmt: dict) -> int:
        """Calculate quality ranking for format sorting."""
        height = fmt.get("height", 0) or 0
        fps = fmt.get("fps", 0) or 0
        filesize = fmt.get("filesize") or fmt.get("filesize_approx") or 0
        
        score = height * 10
        if fps >= 60:
            score += 1000
        elif fps >= 30:
            score += 500
        if 10*1024*1024 <= filesize <= 500*1024*1024:
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
        
        aspect_configs = {
            "16:9": {"width": "560", "height": "315", "padding": "56.25%"},
            "9:16": {"width": "315", "height": "560", "padding": "177.78%"},
            "4:3": {"width": "560", "height": "420", "padding": "75%"},
            "1:1": {"width": "400", "height": "400", "padding": "100%"},
            "21:9": {"width": "560", "height": "240", "padding": "42.86%"}
        }
        
        config = aspect_configs.get(aspect_ratio, aspect_configs["16:9"])
        
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
            
            filters = []
            if options.get("crop_aspect"):
                aspect_filters = {
                    "9:16": "crop=ih*9/16:ih",
                    "16:9": "crop=iw:iw*9/16", 
                    "4:3": "crop=iw:iw*3/4",
                    "1:1": "crop=min(iw\\,ih):min(iw\\,ih)",
                    "21:9": "crop=iw:iw*9/21"
                }
                if options["crop_aspect"] in aspect_filters:
                    filters.append(aspect_filters[options["crop_aspect"]])
            
            if options.get("scale_height"):
                filters.append(f"scale=-2:{options['scale_height']}")
            
            if options.get("fps"):
                filters.append(f"fps={options['fps']}")
            
            if filters:
                cmd.extend(["-vf", ",".join(filters)])
            
            if options.get("video_codec", "h264") == "h265":
                cmd.extend(["-c:v", "libx265", "-preset", "medium"])
            else:
                cmd.extend(["-c:v", "libx264", "-preset", "fast"])
            
            if options.get("audio_codec"):
                cmd.extend(["-c:a", options["audio_codec"]])
            else:
                cmd.extend(["-c:a", "aac"])
            
            if options.get("crf"):
                cmd.extend(["-crf", str(options["crf"])])
            
            cmd.extend(["-f", "mp4", "-y", output_path])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
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
            
            ydl_opts = {
                "format": format_id,
                "outtmpl": os.path.join(output_path, f"video-{video_id}.%(ext)s"),
                "merge_output_format": "mp4",
                "progress_hooks": [progress_hook],
                "quiet": True,
                "no_warnings": True
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_file = os.path.join(output_path, f"video-{info['id']}.mp4")
            
            if not os.path.exists(downloaded_file):
                return None
            
            if processing_options and self.ffmpeg_available:
                processed_file = os.path.join(output_path, f"video-{video_id}-processed.mp4")
                if self.advanced_video_processing(downloaded_file, processed_file, processing_options):
                    os.remove(downloaded_file)
                    return processed_file
            
            return downloaded_file
            
        except Exception as e:
            st.error(f"Download failed: {e}")
            return None

def create_advanced_ui():
    """Create the advanced user interface."""
    st.set_page_config(
        page_title="YouTube Professional Suite",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
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
        
        .video-info-card {
            background: linear-gradient(145deg, #e3f2fd, #bbdefb);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border-left: 4px solid var(--accent-color);
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
        .status-error { background-color: var(--error-color); }
        </style>
    """, unsafe_allow_html=True)

def main():
    create_advanced_ui()
    
    processor = AdvancedYouTubeProcessor()
    
    if 'download_progress' not in st.session_state:
        st.session_state.download_progress = 0.0
    
    st.markdown("""
        <div class="main-header">
            <h1>🎬 YouTube Professional Suite</h1>
            <p>Advanced video processing for single videos</p>
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
    
    with st.sidebar:
        st.header("🔧 Settings")
        
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
        
        with st.expander("🎬 Embed Settings"):
            privacy_enhanced = st.checkbox("🔐 Privacy Enhanced Mode", value=True)
            autoplay = st.checkbox("▶️ Autoplay (muted)")
            show_controls = st.checkbox("🎛️ Show controls", value=True)
            modestbranding = st.checkbox("🏷️ Modest branding")
            loop_video = st.checkbox("🔄 Loop video")
            start_time = st.number_input("⏰ Start time (seconds)", min_value=0, value=0)
            end_time = st.number_input("⏱️ End time (seconds)", min_value=0, value=0)
        
        st.divider()
        system_info = SystemChecker.check_system_resources()
        if "error" not in system_info:
            st.metric("💾 Memory Usage", f"{system_info.get('memory_percent', 0):.1f}%")
            st.metric("💿 Disk Free", f"{system_info.get('disk_free_gb', 0):.1f} GB")
    
    main_col, info_col = st.columns([2.5, 1])
    
    with main_col:
        st.subheader("🎥 Single Video Processing")
        
        url = st.text_input(
            "🔗 YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
            help="Supports all YouTube URL formats including Shorts"
        )
        
        if url:
            video_id = processor.extract_youtube_id(url)
            if video_id:
                st.success(f"✅ Valid YouTube URL detected (ID: {video_id})")
                
                use_cache = st.checkbox("⚡ Use cached data", value=True, 
                                      help="Faster loading for recently analyzed videos")
                
                if st.button("🔍 Analyze Video", use_container_width=True):
                    with st.spinner("🔄 Analyzing video metadata..."):
                        video_info = processor.validate_video(url, use_cache=use_cache)
                    
                    if video_info.get("error"):
                        st.error(f"❌ {video_info['error']}")
                    else:
                        st.session_state.current_video_info = video_info
                        st.success("✅ Video analyzed successfully!")
                
                if hasattr(st.session_state, 'current_video_info'):
                    video_info = st.session_state.current_video_info
                    
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
                    
                    if video_info.get("formats"):
                        st.subheader("📊 Available Formats")
                        
                        format_data = []
                        for fmt in video_info["formats"][:15]:
                            format_data.append({
                                "Quality": fmt.get("quality", "unknown"),
                                "Resolution": fmt.get("resolution", "0x0"),
                                "FPS": fmt.get("fps", 0),
                                "Size (MB)": fmt.get("filesize_mb", 0),
                                "Codec": fmt.get("vcodec", "unknown"),
                                "Format ID": fmt.get("format_id", "")
                            })
                        
                        st.dataframe(format_data, use_container_width=True)
                        
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
                    
                    st.markdown("**Preview:**")
                    st.markdown(embed_codes["responsive"], unsafe_allow_html=True)
                    
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
                    
                    st.subheader("🚀 Actions")
                    
                    col_btn1, col_btn2, col_btn3 = st.columns(3)
                    
                    with col_btn1:
                        if st.button("⬇️ Download Video", use_container_width=True):
                            processing_options = {
                                "crop_aspect": aspect if enable_cropping else None,
                                "video_codec": video_codec.split(" ")[0],
                                "scale_height": CONFIG['QUALITY_PRESETS'].get(quality_preset, "720p").replace("p", "") if enable_optimization else None,
                                "crf": 23 if enable_optimization else None
                            }
                            
                            with st.spinner("📥 Downloading and processing..."):
                                progress_placeholder = st.empty()
                                progress_placeholder.progress(st.session_state.download_progress)
                                
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
                                    
                                    try:
                                        os.remove(file_path)
                                    except:
                                        pass
                                else:
                                    st.error("❌ Download failed!")
                    
                    with col_btn2:
                        if st.button("🎵 Extract Audio", use_container_width=True):
                            with st.spinner("🎵 Extracting audio..."):
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
    
    with info_col:
        st.markdown("""
            <div class="metric-card">
                <h3>📊 System Status</h3>
                <div><span class="status-indicator status-online"></span>Service Online</div>
                <div><span class="status-indicator {}"></span>FFmpeg {}</div>
                <div><span class="status-indicator status-online"></span>Cache Active</div>
            </div>
        """.format(
            "status-online" if processor.ffmpeg_available else "status-error",
            "Ready" if processor.ffmpeg_available else "Missing"
        ), unsafe_allow_html=True)
    
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