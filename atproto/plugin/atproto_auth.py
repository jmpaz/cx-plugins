from __future__ import annotations

import base64
import json
import os

import click

from contextualize.auth.common import (
    load_dotenv_optional,
    open_url_in_browser,
    should_confirm_login,
)


def _atproto_session_profile(
    access_token: str,
    *,
    resource_server: str | None = None,
    dpop_private_key_pem: str | None = None,
    dpop_public_jwk: dict[str, str] | None = None,
    token_type: str = "Bearer",
    resource_nonce: str | None = None,
) -> dict[str, str] | None:
    import requests

    host = (
        (resource_server or "").strip()
        or (os.environ.get("ATPROTO_PDS_HOST") or "").strip()
        or "https://bsky.social"
    ).rstrip("/")
    url = f"{host}/xrpc/com.atproto.server.getSession"
    if (
        dpop_private_key_pem
        and dpop_public_jwk
        and token_type.strip().lower() == "dpop"
    ):
        from contextualize.references.atproto_auth import AtprotoDPoPAuth

        current_nonce = (resource_nonce or "").strip() or None
        for _ in range(2):
            auth = AtprotoDPoPAuth(
                access_token=access_token,
                private_key_pem=dpop_private_key_pem,
                public_jwk=dpop_public_jwk,
                token_type=token_type,
                resource_nonce=current_nonce,
            )
            response = requests.get(
                url, headers={"Accept": "application/json"}, auth=auth, timeout=15
            )
            next_nonce = (
                response.headers.get("DPoP-Nonce") or ""
            ).strip() or auth.resource_nonce
            if response.ok:
                payload = response.json()
                if not isinstance(payload, dict):
                    return None
                handle = payload.get("handle")
                did = payload.get("did")
                return {
                    "handle": handle.strip() if isinstance(handle, str) else "",
                    "did": did.strip() if isinstance(did, str) else "",
                }
            if (
                response.status_code in {400, 401}
                and next_nonce
                and next_nonce != current_nonce
            ):
                current_nonce = next_nonce
                continue
            response.raise_for_status()
        raise RuntimeError("ATProto session probe failed after nonce retry")
    response = requests.get(
        url,
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
    handle = payload.get("handle")
    did = payload.get("did")
    return {
        "handle": handle.strip() if isinstance(handle, str) else "",
        "did": did.strip() if isinstance(did, str) else "",
    }


def _atproto_profile_from_did(did: str) -> dict[str, str] | None:
    import requests

    cleaned = did.strip()
    if not cleaned:
        return None
    public_host = (
        os.environ.get("ATPROTO_PUBLIC_HOST") or ""
    ).strip() or "https://public.api.bsky.app"
    public_host = public_host.rstrip("/")
    response = requests.get(
        f"{public_host}/xrpc/app.bsky.actor.getProfile",
        headers={"Accept": "application/json"},
        params={"actor": cleaned},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    handle = payload.get("handle")
    did_value = payload.get("did")
    return {
        "handle": handle.strip() if isinstance(handle, str) else "",
        "did": did_value.strip() if isinstance(did_value, str) else cleaned,
    }


def _atproto_subject_from_access_token(access_token: str) -> str | None:
    parts = access_token.strip().split(".")
    if len(parts) != 3:
        return None
    payload_part = parts[1]
    padding = "=" * (-len(payload_part) % 4)
    try:
        decoded = base64.urlsafe_b64decode((payload_part + padding).encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    subject = payload.get("sub")
    if not isinstance(subject, str):
        return None
    cleaned = subject.strip()
    return cleaned or None


def _atproto_refresh_cached_oauth_session() -> dict[str, object] | None:
    try:
        from contextualize.references.atproto import _refresh_cached_oauth_session

        refreshed = _refresh_cached_oauth_session()
        return refreshed if isinstance(refreshed, dict) else None
    except Exception:
        return None


def _atproto_has_active_user_auth() -> bool:
    from contextualize.cache.atproto import get_cached_oauth_access_token

    load_dotenv_optional()
    env_token = (os.environ.get("ATPROTO_ACCESS_TOKEN") or "").strip()
    if env_token:
        return True
    refreshed = _atproto_refresh_cached_oauth_session()
    if isinstance(refreshed, dict):
        refreshed_access = str(refreshed.get("access_token") or "").strip()
        if refreshed_access:
            return True
    return bool(get_cached_oauth_access_token(min_valid_seconds=60))


def _print_atproto_auth_status() -> None:
    from contextualize.cache.atproto import (
        get_cached_oauth_access_token,
        get_cached_oauth_session_record,
    )

    load_dotenv_optional()
    env_access = (os.environ.get("ATPROTO_ACCESS_TOKEN") or "").strip()
    env_refresh = (os.environ.get("ATPROTO_REFRESH_TOKEN") or "").strip()
    refreshed_record = _atproto_refresh_cached_oauth_session()
    record = (
        refreshed_record
        if isinstance(refreshed_record, dict)
        else get_cached_oauth_session_record()
    )
    cached_access = get_cached_oauth_access_token(min_valid_seconds=60)
    if not cached_access and isinstance(refreshed_record, dict):
        refreshed_access = str(refreshed_record.get("access_token") or "").strip()
        cached_access = refreshed_access or None

    click.echo("ATProto auth status")
    click.echo("-------------------")
    click.echo(f"env access token: {'present' if bool(env_access) else 'not set'}")
    click.echo(f"env refresh token: {'present' if bool(env_refresh) else 'not set'}")
    click.echo(
        f"stored oauth session: {'present' if record is not None else 'not set'}"
    )
    if record is not None:
        expires_at = record.get("expires_at")
        refresh_token = record.get("refresh_token")
        client_id = record.get("client_id")
        click.echo(
            f"stored oauth token valid now: {'yes' if bool(cached_access) else 'no'}"
        )
        if isinstance(expires_at, str) and expires_at:
            click.echo(f"stored token expires_at: {expires_at}")
        if isinstance(client_id, str) and client_id:
            click.echo(f"stored client_id: {client_id}")
        click.echo(
            f"stored refresh token: {'present' if isinstance(refresh_token, str) and bool(refresh_token.strip()) else 'not set'}"
        )

    active_token = env_access or cached_access
    if active_token:
        source = "env access token" if env_access else "stored oauth token"
        click.echo(f"active auth source: {source}")
        profile = None
        probe_error: Exception | None = None
        subject_did = None
        if isinstance(record, dict):
            subject_did = str(record.get("subject_did") or "").strip() or None
        if not subject_did:
            subject_did = _atproto_subject_from_access_token(active_token)
        if subject_did:
            try:
                profile = _atproto_profile_from_did(subject_did)
            except Exception:
                profile = None
        try:
            if env_access and not profile:
                profile = _atproto_session_profile(active_token)
            elif isinstance(record, dict) and not profile:
                dpop_public_jwk = record.get("dpop_public_jwk")
                profile = _atproto_session_profile(
                    active_token,
                    resource_server=str(record.get("resource_server") or "").strip()
                    or None,
                    dpop_private_key_pem=str(record.get("dpop_private_key_pem") or "")
                    or None,
                    dpop_public_jwk=dpop_public_jwk
                    if isinstance(dpop_public_jwk, dict)
                    else None,
                    token_type=str(record.get("token_type") or "DPoP"),
                    resource_nonce=str(
                        record.get("resource_server_nonce") or ""
                    ).strip()
                    or None,
                )
        except Exception as exc:
            probe_error = exc
        if profile:
            handle = profile.get("handle") or "(unknown)"
            did = profile.get("did") or ""
            click.echo(f"resolved user: {handle}")
            if did:
                click.echo(f"resolved did: {did}")
        elif probe_error is not None:
            click.echo(f"getSession probe: failed ({probe_error})")
        else:
            click.echo("resolved user: unavailable")
    else:
        click.echo("active auth source: public mode")


def _run_atproto_logout() -> None:
    from contextualize.cache.atproto import clear_cached_oauth_session

    load_dotenv_optional()
    clear_cached_oauth_session()
    click.echo("Cleared stored ATProto OAuth session.")
    if (os.environ.get("ATPROTO_ACCESS_TOKEN") or "").strip():
        click.echo(
            "ATPROTO_ACCESS_TOKEN is still set in environment and will continue to override stored auth."
        )


def _run_atproto_login(timeout: int, no_browser: bool) -> None:
    from contextualize.cache.atproto import store_oauth_session
    from contextualize.references.atproto_auth import (
        build_authorize_url,
        build_pkce_challenge,
        exchange_authorization_code,
        generate_dpop_key_pair,
        generate_pkce_verifier,
        generate_state,
        load_atproto_oauth_config,
        push_authorization_request,
        wait_for_oauth_callback,
    )

    if timeout < 5:
        raise click.BadParameter("--timeout must be at least 5 seconds")

    config = load_atproto_oauth_config()
    verifier = generate_pkce_verifier()
    challenge = build_pkce_challenge(verifier)
    state = generate_state()
    key_pair = generate_dpop_key_pair()
    request_uri, auth_server_nonce = push_authorization_request(
        config,
        code_challenge=challenge,
        state=state,
        key_pair=key_pair,
    )
    authorize_url = build_authorize_url(config, request_uri=request_uri)

    click.echo("Starting ATProto sign-in.")
    click.echo("If your browser does not open automatically, open this URL manually:")
    click.echo(authorize_url)

    if not no_browser and not open_url_in_browser(authorize_url):
        click.echo("Could not open browser automatically.", err=True)

    callback_result = wait_for_oauth_callback(
        config,
        expected_state=state,
        timeout_seconds=timeout,
    )
    token, auth_server_nonce = exchange_authorization_code(
        config,
        code=callback_result.code,
        code_verifier=verifier,
        key_pair=key_pair,
        auth_server_nonce=auth_server_nonce,
    )
    store_oauth_session(
        access_token=token.access_token,
        refresh_token=token.refresh_token,
        expires_in_seconds=token.expires_in,
        token_type=token.token_type,
        scope=token.scope,
        client_id=config.client_id,
        auth_server=config.authorization_server,
        resource_server=config.resource_server,
        dpop_private_key_pem=key_pair.private_key_pem,
        dpop_public_jwk=key_pair.public_jwk,
        auth_server_nonce=auth_server_nonce,
        resource_server_nonce=None,
        subject_did=token.subject_did,
    )

    profile = None
    try:
        profile = _atproto_session_profile(
            token.access_token,
            resource_server=config.resource_server,
            dpop_private_key_pem=key_pair.private_key_pem,
            dpop_public_jwk=key_pair.public_jwk,
            token_type=token.token_type,
        )
    except Exception:
        profile = None
    if not profile and token.subject_did:
        try:
            profile = _atproto_profile_from_did(token.subject_did)
        except Exception:
            profile = None

    click.echo("ATProto login successful.")
    click.echo(f"Stored oauth token (expires in ~{token.expires_in // 60} minutes).")
    if profile:
        handle = profile.get("handle") or "(unknown)"
        did = profile.get("did") or ""
        click.echo(f"Signed in as: {handle}")
        if did:
            click.echo(f"User DID: {did}")


def auth_atproto(logout_requested: bool, timeout: int, no_browser: bool) -> None:
    if logout_requested:
        _run_atproto_logout()
        return

    if _atproto_has_active_user_auth():
        _print_atproto_auth_status()
        return

    if not should_confirm_login("ATProto"):
        click.echo("ATProto sign-in canceled.")
        return

    try:
        _run_atproto_login(timeout=timeout, no_browser=no_browser)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"ATProto login failed: {exc}") from exc


def register_auth_command(group: click.Group) -> None:
    @group.command("atproto")
    @click.option(
        "--logout",
        "logout_requested",
        is_flag=True,
        help="Clear stored ATProto OAuth session.",
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
    def _auth_atproto(
        logout_requested: bool,
        timeout: int,
        no_browser: bool,
    ) -> None:
        auth_atproto(
            logout_requested=logout_requested,
            timeout=timeout,
            no_browser=no_browser,
        )
