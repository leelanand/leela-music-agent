"""Upload lo-fi video to the Daily Lofi Beats YouTube channel."""

import time
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

SCOPES             = ["https://www.googleapis.com/auth/youtube.force-ssl"]
CLIENT_SECRET_FILE = Path(__file__).parent / "client_secret.json"
TOKEN_FILE         = Path(__file__).parent / "token_music.json"


def _get_credentials() -> Credentials:
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CLIENT_SECRET_FILE.exists():
                raise FileNotFoundError(
                    "client_secret.json not found — copy it from the YouTube agent folder.\n"
                    f"Expected: {CLIENT_SECRET_FILE}"
                )
            flow  = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    return creds


def upload_video(video_path: Path, title: str, description: str,
                 tags: list[str], thumbnail_path: Path | None = None) -> str:
    print("[5/5] Uploading to YouTube...")
    creds   = _get_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title":       title[:100],
            "description": description,
            "tags":        tags,
            "categoryId":  "10",   # 10 = Music
        },
        "status": {"privacyStatus": "public"},
    }

    media   = MediaFileUpload(str(video_path), mimetype="video/mp4",
                              resumable=True, chunksize=1024 * 1024)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                print(f"   Progress: {int(status.progress() * 100)}%")
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504):
                time.sleep(5)
            else:
                raise

    video_id  = response["id"]
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"   Published: {video_url}")

    if thumbnail_path and thumbnail_path.exists():
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg"),
        ).execute()
        print("   Thumbnail uploaded.")

    return video_url
