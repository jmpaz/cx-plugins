from __future__ import annotations

import os

import click

from contextualize.auth.common import (
    accent_url,
    is_tty_session,
    load_dotenv_optional,
    open_url_in_browser,
)


def _arena_pat_setup_url() -> str:
    return "https://www.are.na/settings/personal-access-tokens"


def _arena_me_profile(access_token: str) -> dict[str, str] | None:
    import requests

    response = requests.get(
        "https://api.are.na/v3/me",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    username = payload.get("username") or payload.get("name")
    slug = payload.get("slug")
    return {
        "username": username.strip() if isinstance(username, str) else "",
        "slug": slug.strip() if isinstance(slug, str) else "",
    }


def _resolve_arena_active_token() -> tuple[str | None, str]:
    from contextualize.cache.arena import get_cached_user_access_token

    load_dotenv_optional()
    env_token = (os.environ.get("ARENA_ACCESS_TOKEN") or "").strip()
    if env_token:
        return env_token, "env user token"
    cached = get_cached_user_access_token(min_valid_seconds=60)
    if cached:
        return cached, "stored user token"
    return None, "none"


def _probe_arena_token(
    access_token: str,
) -> tuple[dict[str, str] | None, int | None, str]:
    try:
        return _arena_me_profile(access_token), None, ""
    except Exception as exc:
        status_code = None
        response = getattr(exc, "response", None)
        if response is not None:
            candidate = getattr(response, "status_code", None)
            if isinstance(candidate, int):
                status_code = candidate
        return None, status_code, str(exc)


def _arena_has_active_user_auth() -> bool:
    token, _source = _resolve_arena_active_token()
    return bool(token)


def _print_arena_auth_status() -> None:
    from contextualize.cache.arena import (
        get_cached_user_access_token,
        get_cached_user_token_record,
    )

    load_dotenv_optional()
    env_token = (os.environ.get("ARENA_ACCESS_TOKEN") or "").strip()
    record = get_cached_user_token_record()
    cached_access = get_cached_user_access_token(min_valid_seconds=60)

    click.echo("Are.na auth status")
    click.echo("------------------")
    click.echo(f"env user token: {'present' if bool(env_token) else 'not set'}")
    click.echo(f"stored user token: {'present' if record is not None else 'not set'}")
    if record is not None:
        expires_at = record.get("expires_at")
        click.echo(
            f"stored user token unexpired: {'yes' if bool(cached_access) else 'no'}"
        )
        if isinstance(expires_at, str) and expires_at:
            click.echo(f"stored token expires_at: {expires_at}")

    active_token, source = _resolve_arena_active_token()
    if active_token:
        click.echo(f"active auth source: {source}")
        profile, status_code, _error_message = _probe_arena_token(active_token)
        if profile:
            username = profile.get("username") or "(unknown)"
            click.echo(f"resolved user: {username}")
        else:
            click.echo("resolved user: unavailable")
            if status_code == 401:
                if source == "stored user token":
                    click.echo(
                        "hint: run `contextualize auth arena --logout` then `contextualize auth arena`"
                    )
                else:
                    click.echo("hint: update or unset ARENA_ACCESS_TOKEN, then retry")
    else:
        click.echo("active auth source: none")


def _validate_arena_token_for_storage(token: str) -> dict[str, str]:
    token_clean = token.strip()
    if not token_clean:
        raise click.BadParameter("--token cannot be empty")
    profile, status_code, _error_message = _probe_arena_token(token_clean)
    if profile:
        return profile
    if status_code == 401:
        raise click.ClickException("Are.na token rejected (401 Unauthorized).")
    raise click.ClickException("Could not verify Are.na token right now.")


def _run_arena_logout() -> None:
    from contextualize.cache.arena import clear_cached_user_token

    load_dotenv_optional()
    clear_cached_user_token()
    click.echo("Cleared stored Are.na user token.")
    if (os.environ.get("ARENA_ACCESS_TOKEN") or "").strip():
        click.echo(
            "ARENA_ACCESS_TOKEN is still set in environment and will continue to override stored auth."
        )


def _store_arena_pat(token: str, *, profile: dict[str, str] | None = None) -> None:
    from contextualize.cache.arena import store_user_token

    token_clean = token.strip()
    if not token_clean:
        raise click.BadParameter("--token cannot be empty")
    store_user_token(
        access_token=token_clean,
        refresh_token=None,
        expires_in_seconds=31536000,
        token_type="Bearer",
        scope="",
    )

    click.echo("Are.na token stored.")
    if profile:
        username = profile.get("username") or "(unknown)"
        click.echo(f"Signed in as: {username}")


def _prompt_arena_pat_interactive() -> tuple[str, dict[str, str]] | None:
    def _mask_token(value: str) -> str:
        if not value:
            return ""
        if len(value) <= 4:
            return "*" * len(value)
        if len(value) <= 12:
            return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"
        return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"

    while True:
        pasted = click.prompt(
            "Paste personal access token",
            default="",
            show_default=False,
            hide_input=True,
        ).strip()
        if not pasted:
            if click.confirm("No token entered. Try again?", default=True):
                continue
            return None

        click.echo(f"Entered token: {_mask_token(pasted)}")
        profile, status_code, _error_message = _probe_arena_token(pasted)
        if not profile:
            if status_code == 401:
                click.secho("Token rejected by Are.na.", fg="yellow")
            else:
                click.secho("Could not verify token right now.", fg="yellow")
            if click.confirm("Try again?", default=True):
                continue
            return None

        username = profile.get("username") or "(unknown)"
        click.echo(f"resolved user: {username}")
        if click.confirm("Save this token?", default=True):
            return pasted, profile
        if click.confirm("Re-enter token?", default=True):
            continue
        return None


def _maybe_open_arena_token_page_interactive() -> None:
    url = _arena_pat_setup_url()
    display_url = accent_url(url)
    open_now = click.confirm(
        "Open the Are.na token page now?",
        default=True,
    )
    if open_now:
        if open_url_in_browser(url):
            click.secho("Opened in browser.", fg="green")
        else:
            click.secho("Could not open browser automatically.", fg="yellow")
        click.echo(f"URL: {display_url}")
        return
    click.echo(f"URL: {display_url}")


def auth_arena(
    logout_requested: bool,
    token: str | None,
) -> None:
    if logout_requested:
        _run_arena_logout()
        return

    if token is not None:
        profile = _validate_arena_token_for_storage(token.strip())
        _store_arena_pat(token, profile=profile)
        return

    if _arena_has_active_user_auth():
        _print_arena_auth_status()
        return

    url = _arena_pat_setup_url()
    display_url = accent_url(url)
    if not is_tty_session():
        click.echo(
            "contextualize needs an Are.na personal access token to access private blocks & channels."
        )
        click.echo(f"URL: {display_url}")
        click.echo(
            "Create a read-only token, then pass --token or set ARENA_ACCESS_TOKEN."
        )
        raise click.ClickException(
            "No interactive TTY. Pass --token or set ARENA_ACCESS_TOKEN."
        )

    click.echo(
        "contextualize needs an Are.na personal access token to access private blocks & channels."
    )
    click.echo()
    _maybe_open_arena_token_page_interactive()
    click.echo()
    click.echo("Create a read-only token on that page.")
    click.echo()

    result = _prompt_arena_pat_interactive()
    if not result:
        click.echo("Are.na token setup canceled.")
        return

    pasted, profile = result
    _store_arena_pat(pasted, profile=profile)


def register_auth_command(group: click.Group) -> None:
    @group.command("arena")
    @click.option(
        "--logout",
        "logout_requested",
        is_flag=True,
        help="Clear stored Are.na login.",
    )
    @click.option(
        "--token",
        type=str,
        default=None,
        help="Store an Are.na personal access token.",
    )
    def _auth_arena(
        logout_requested: bool,
        token: str | None,
    ) -> None:
        auth_arena(
            logout_requested=logout_requested,
            token=token,
        )
