"""YouTube comment extraction and batch toxicity analysis for HateShield."""
import itertools
import re

from youtube_comment_downloader import SORT_BY_POPULAR, YoutubeCommentDownloader

from utils import predecir_toxicidad

# Matches the 11-character video ID across the common YouTube URL formats:
# watch?v=..., youtu.be/..., /embed/..., /shorts/...
_VIDEO_ID_PATTERN = r"(?:v=|youtu\.be/|/embed/|/shorts/)([0-9A-Za-z_-]{11})"


def extract_video_id(url: str) -> str | None:
    """Extract the 11-character YouTube video ID from a video URL.

    Supports standard watch URLs, short youtu.be links, /embed/ and /shorts/
    links. Returns None if no valid video ID is found.
    """
    match = re.search(_VIDEO_ID_PATTERN, url)
    return match.group(1) if match else None


def fetch_comments(url: str, limit: int = 50) -> list[str]:
    """Download up to `limit` comments from a YouTube video URL.

    Raises ValueError if the URL does not contain a valid video ID.
    """
    video_id = extract_video_id(url)
    if video_id is None:
        raise ValueError(f"Could not extract a video ID from URL: {url}")

    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments(video_id, sort_by=SORT_BY_POPULAR)
    return [comment["text"] for comment in itertools.islice(comments, limit)]


def analyze_youtube_video(url: str, ensemble, tfidf, limit: int = 50) -> list[dict]:
    """Fetch comments from a YouTube video and run toxicity prediction on each.

    Returns a list of dicts, one per comment, combining the original comment
    text with the output of predecir_toxicidad (es_toxico, votos_toxico,
    total_modelos).
    """
    comments = fetch_comments(url, limit=limit)
    results = []
    for comment in comments:
        prediction = predecir_toxicidad(comment, ensemble, tfidf)
        results.append({"comentario": comment, **prediction})
    return results
