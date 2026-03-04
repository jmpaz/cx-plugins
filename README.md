# cx-plugins

External provider plugins for `contextualize`.

This package exposes plugins through the `contextualize.plugins` entrypoint group.

Included providers:
- arena
- atproto
- discord
- soundcloud
- youtube

## Install

Install via `contextualize` optional dependency:

```bash
uv pip install "contextualize[plugins]"
```

## Local development

Filesystem plugins may also be loaded directly by pointing
`CONTEXTUALIZE_PLUGIN_DIRS` at directories that contain `plugin.yaml` manifests.
