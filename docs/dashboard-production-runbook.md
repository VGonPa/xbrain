# Dashboard Production Runbook

This runbook is for operating the served XBrain dashboard from `main`. The
current production dashboard is the forest light/dark workspace UI served behind
Basic Auth at `brainxterminal.duckdns.org`.

## Scope

- Source template: `src/xbrain/resources/dashboard.template.html`.
- Generated artifact: `<vault>/<output_subdir>/dashboard.html`.
- Local server: `uv run xbrain serve-dashboard --host 127.0.0.1 --port 8765`.
- Hardened examples: `deploy/systemd/xbrain-dashboard.service.example` and
  `deploy/nginx/xbrain-dashboard.conf.example`.
- Current production service: `/etc/systemd/system/xbrain-dashboard.service`.
- Current public proxy: `/etc/nginx/sites-available/brainxterminal`, with
  `brainxterminal.duckdns.org` protected by Basic Auth and proxied to
  `http://127.0.0.1:8765`.

The served dashboard exposes full-note search, Ask, and operational actions.
Do not bind it directly to an unauthenticated public interface. Prefer
`127.0.0.1` plus TLS and authenticated reverse-proxy access. The application
adds CSP, clickjacking/MIME/referrer/permissions headers; verifies same-origin
custom-header requests for Ask and every mutation; limits request bodies; and
rate-limits search, Ask, and mutations. Proxy authentication and limits remain
the outer layer, not a replacement.

## Release Steps

1. Check out the production branch and pull the latest GitHub state.

   ```bash
   git switch main
   git pull --ff-only origin main
   ```

2. Generate the latest dashboard artifact from the current XBrain data.

   ```bash
   uv run xbrain generate
   ```

3. Restart the local served dashboard.

   ```bash
   systemctl restart xbrain-dashboard.service
   ```

4. Confirm the production reverse proxy still points at
   `http://127.0.0.1:8765`.

   ```bash
   nginx -T | grep -A20 brainxterminal.duckdns.org
   ```

5. Open the protected production URL and verify these routes:

   - `/#library`: search, filters, sort, quick views, and document drawer.
   - `/#atlas`: mind map first, Timeline, Documents, and topic explorer.
   - `/#analytics`: cross-filtered KPIs, charts, and data-table alternatives.
   - `/#ops`: refresh/retry, failure evidence, commands, and storage.
   - Ask button: global drawer, current-selection scope, answer evidence, and
     keyboard focus containment.

## Smoke Checks

Run these on the production host:

```bash
curl -I http://127.0.0.1:8765/dashboard.html
curl http://127.0.0.1:8765/api/status
curl --get --data-urlencode 'q=agent workflows' http://127.0.0.1:8765/api/search
curl -I https://brainxterminal.duckdns.org
```

Expected result: the dashboard request returns `200`, and `/api/status` returns
JSON with pipeline state fields. If the public URL is intentionally protected,
an unauthenticated outside request may return `401`; verify through the
authorized path instead of disabling protection. A protected public response
should include `WWW-Authenticate: Basic realm="BrainX Dashboard"`.

Confirm the local dashboard response also contains `Content-Security-Policy`,
`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and
`Cache-Control: no-store`. A mutating API smoke request must use both the local
origin and the explicit action header:

```bash
curl -i -X POST http://127.0.0.1:8765/api/retry-failed \
  -H 'Origin: http://127.0.0.1:8765' \
  -H 'X-XBrain-Action: 1'
```

Omitting the action header or supplying a foreign `Origin` should return `403`.
Repeated expensive requests should eventually return `429` with `Retry-After`.

## Hardened Service and Proxy

Copy the examples and replace every path, user, hostname, certificate, vault,
and environment-file location before enabling them:

```bash
install -o root -g root -m 0644 \
  deploy/systemd/xbrain-dashboard.service.example \
  /etc/systemd/system/xbrain-dashboard.service
install -o root -g root -m 0644 \
  deploy/nginx/xbrain-dashboard.conf.example \
  /etc/nginx/sites-available/xbrain-dashboard
```

Run the service as the dedicated unprivileged `xbrain` account. Keep secrets in
`/etc/xbrain/dashboard.env` with mode `0600`; never put them in the unit or
repository. The example uses a read-only system view, private temp/devices,
an empty capability set, and explicit writable data/vault paths. Create those
paths before starting the unit. Its `XBRAIN_REPO_ROOT=/var/lib/xbrain` means a
production `config.toml` must also exist there and point its vault at the
allowed `/srv/xbrain-vault` path (or at the Drive cache beneath `/var/lib/xbrain`).

The Nginx example requires Basic Auth, TLS, small request bodies, and separate
rate zones for search, Ask, and actions. Its `limit_req_zone` directives belong
inside Nginx's top-level `http {}` block. Validate both layers before reload:

```bash
systemd-analyze verify /etc/systemd/system/xbrain-dashboard.service
nginx -t
systemctl daemon-reload
systemctl enable --now xbrain-dashboard.service
systemctl reload nginx
```

## Validation Evidence

Before restarting production after code changes, run:

```bash
uv run pytest tests/test_dashboard.py -q
uv run pytest -q
```

The dashboard test suite covers shared filters, full Library rendering,
workspace routing, accessible drawers/tabs, chart alternatives, reduced motion,
mobile overflow/touch targets, Atlas Documents, metric semantics, and graph
layout stability. HTTP tests cover origin/header enforcement, security headers,
rate limiting and full-note search; Drive/fetch tests cover upload conflicts,
failed-command uploads and private-network article URLs.
Use a browser or Playwright against the served dashboard for final visual checks
at desktop and mobile widths. Include `/notes?path=...` in visual checks because
the web note viewer is served by the same process.

See `docs/atlas-knowledge-graph.md` for the graph data contract, controls,
local layout persistence, and served topic-action behavior.

## Rollback

1. Switch back to the previous known-good commit or tag on `main`.
2. Regenerate the dashboard with `uv run xbrain generate`.
3. Restart `xbrain-dashboard.service` and reload Nginx if proxy settings changed.
4. Re-run the smoke checks above.
