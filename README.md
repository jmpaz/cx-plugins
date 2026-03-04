# cx-plugins

[`contextualize`](https://github.com/jmpaz/contextualize) plugins resolve supported targets to plaintext.

This package provides the following plugins:

| Plugin | Provider | Handles |
| --- | --- | --- |
| `arena` | [Are.na](https://www.are.na/) | blocks, channels |
| `atproto` | [AT Protocol](https://atproto.com/docs) | posts, profiles, lists, starter packs |
| `discord` | Discord | messages, threads, channels |
| `soundcloud` | SoundCloud | tracks, playlists, profiles |
| `youtube` | YouTube | videos |

## Installation

Install via `contextualize` optional dependency:

```bash
uv pip install "contextualize[plugins]"
```

## Local development

Filesystem plugins may also be loaded directly by pointing
`CONTEXTUALIZE_PLUGIN_DIRS` at directories that contain `plugin.yaml` manifests.
