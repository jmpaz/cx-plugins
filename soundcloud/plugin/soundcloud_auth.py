from __future__ import annotations

import os

import click

from contextualize.auth.common import (
    load_dotenv_optional,
    should_confirm_login,
    open_url_in_browser,
)


def _soundcloud_me_profile(access_token: str) -> dict[str, str] | None:
    import requests

    response = requests.get(
        "https://api.soundcloud.com/me",
        headers={
            "accept": "application/json; charset=utf-8",
            "Authorization": f"Bearer {access_token}",
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    username = payload.get("username")
    urn = payload.get("urn")
    permalink = payload.get("permalink")
    return {
        "username": username.strip() if isinstance(username, str) else "",
        "urn": urn.strip() if isinstance(urn, str) else "",
        "permalink": permalink.strip() if isinstance(permalink, str) else "",
    }


def _run_soundcloud_login(timeout: int, no_browser: bool) -> None:
    from contextualize.cache.soundcloud import store_user_token
    from contextualize.references.soundcloud_auth import (
        build_authorize_url,
        build_pkce_challenge,
        exchange_authorization_code,
        generate_pkce_verifier,
        generate_state,
        load_soundcloud_oauth_config,
        wait_for_oauth_callback,
    )

    if timeout < 5:
        raise click.BadParameter("--timeout must be at least 5 seconds")

    config = load_soundcloud_oauth_config()
    verifier = generate_pkce_verifier()
    challenge = build_pkce_challenge(verifier)
    state = generate_state()
    authorize_url = build_authorize_url(
        config,
        code_challenge=challenge,
        state=state,
    )

    click.echo("Starting SoundCloud sign-in.")
    click.echo(f"Redirect URI: {config.redirect_uri}")
    click.echo("If your browser does not open automatically, open this URL manually:")
    click.echo(authorize_url)

    if not no_browser and not open_url_in_browser(authorize_url):
        click.echo("Could not open browser automatically.", err=True)

    callback_result = wait_for_oauth_callback(
        config,
        expected_state=state,
        timeout_seconds=timeout,
    )
    token = exchange_authorization_code(
        config,
        code=callback_result.code,
        code_verifier=verifier,
    )
    store_user_token(
        access_token=token.access_token,
        refresh_token=token.refresh_token,
        expires_in_seconds=token.expires_in,
        token_type=token.token_type,
        scope=token.scope,
    )

    profile = None
    try:
        profile = _soundcloud_me_profile(token.access_token)
    except Exception:
        profile = None

    click.echo("SoundCloud login successful.")
    click.echo(f"Stored user token (expires in ~{token.expires_in // 60} minutes).")
    if profile:
        username = profile.get("username") or "(unknown)"
        urn = profile.get("urn") or ""
        click.echo(f"Signed in as: {username}")
        if urn:
            click.echo(f"User URN: {urn}")


def _soundcloud_has_active_user_auth() -> bool:
    from contextualize.cache.soundcloud import get_cached_user_access_token

    load_dotenv_optional()
    env_token = (os.environ.get("SOUNDCLOUD_ACCESS_TOKEN") or "").strip()
    if env_token:
        return True
    return bool(get_cached_user_access_token(min_valid_seconds=60))


def _print_soundcloud_auth_status() -> None:
    from contextualize.cache.soundcloud import (
        get_cached_user_access_token,
        get_cached_user_token_record,
    )
    from contextualize.references.soundcloud_auth import (
        load_soundcloud_client_credentials,
    )

    load_dotenv_optional()

    env_token = (os.environ.get("SOUNDCLOUD_ACCESS_TOKEN") or "").strip()
    record = get_cached_user_token_record()
    cached_access = get_cached_user_access_token(min_valid_seconds=60)

    click.echo("SoundCloud auth status")
    click.echo("---------------------")
    click.echo(f"env user token: {'present' if bool(env_token) else 'not set'}")
    click.echo(f"stored user token: {'present' if record is not None else 'not set'}")
    if record is not None:
        expires_at = record.get("expires_at")
        refresh_token = record.get("refresh_token")
        click.echo(
            f"stored user token valid now: {'yes' if bool(cached_access) else 'no'}"
        )
        if isinstance(expires_at, str) and expires_at:
            click.echo(f"stored token expires_at: {expires_at}")
        click.echo(
            f"stored refresh token: {'present' if isinstance(refresh_token, str) and bool(refresh_token.strip()) else 'not set'}"
        )

    try:
        load_soundcloud_client_credentials()
        click.echo("client credentials: configured")
    except Exception:
        click.echo("client credentials: not configured")

    active_token = env_token or cached_access
    if active_token:
        source = "env user token" if env_token else "stored user token"
        click.echo(f"active auth source: {source}")
        try:
            profile = _soundcloud_me_profile(active_token)
        except Exception as exc:
            click.echo(f"/me probe: failed ({exc})")
        else:
            if profile:
                click.echo("/me probe: ok")
                username = profile.get("username") or "(unknown)"
                click.echo(f"resolved user: {username}")
            else:
                click.echo("/me probe: no profile payload")
    else:
        click.echo("active auth source: app token or public mode")


def _run_soundcloud_logout() -> None:
    from contextualize.cache.soundcloud import clear_cached_user_token

    load_dotenv_optional()

    clear_cached_user_token()
    click.echo("Cleared stored SoundCloud user token.")
    if (os.environ.get("SOUNDCLOUD_ACCESS_TOKEN") or "").strip():
        click.echo(
            "SOUNDCLOUD_ACCESS_TOKEN is still set in environment and will continue to override stored auth."
        )


def auth_soundcloud(
    logout_requested: bool,
    timeout: int,
    no_browser: bool,
) -> None:
    if logout_requested:
        _run_soundcloud_logout()
        return

    if _soundcloud_has_active_user_auth():
        _print_soundcloud_auth_status()
        return

    if not should_confirm_login("SoundCloud"):
        click.echo("SoundCloud sign-in canceled.")
        return

    _run_soundcloud_login(timeout=timeout, no_browser=no_browser)


def register_auth_command(group: click.Group) -> None:
    @group.command("soundcloud")
    @click.option(
        "--logout",
        "logout_requested",
        is_flag=True,
        help="Clear stored SoundCloud login.",
    )
    @click.option(
        "--timeout",
        type=int,
        default=180,
        show_default=True,
        help="Seconds to wait for the OAuth callback.",
    )
    @click.option(
        "--no-browser",
        is_flag=True,
        help="Do not auto-open the authorization URL in a browser.",
    )
    def _auth_soundcloud(
        logout_requested: bool,
        timeout: int,
        no_browser: bool,
    ) -> None:
        auth_soundcloud(
            logout_requested=logout_requested,
            timeout=timeout,
            no_browser=no_browser,
        )
