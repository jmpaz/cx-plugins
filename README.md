# cx-plugins

[`contextualize`](https://github.com/jmpaz/contextualize) plugins resolve supported targets to plaintext.

This package provides the following plugins:

| Plugin | Provider | Handles |
| --- | --- | --- |
| `arena` | [Are.na](https://www.are.na/) | blocks, channels |
| `arxiv` | [arXiv](https://arxiv.org/) | papers |
| `atproto` | [AT Protocol](https://atproto.com/docs) | posts, profiles, lists, starter packs |
| `discord` | Discord | messages, threads, channels |
| `soundcloud` | SoundCloud | tracks, playlists, profiles |
| `wikipedia` | Wikipedia | article pages |
| `ytdlp` | [yt-dlp](https://github.com/yt-dlp/yt-dlp) | supported media URLs (YouTube/Vimeo, TikTok/IG, etc)|

## Installation

Install via `contextualize` optional dependency:

```bash
uv pip install "contextualize[plugins]"
```
