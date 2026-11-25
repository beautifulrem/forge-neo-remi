# Security

How the Remi WebUI protects itself and its user: the threat model, the
controls and where they live, how to run the server where others can reach
it, and what remains out of scope. The controls are tested in
`tests/test_remi_security.py`, `tests/test_remi_server_auth.py`,
`tests/test_extension_route_guard.py`, `tests/test_remi_server_files.py`,
`tests/test_remi_server_runtime.py` (startup makes no outbound
connection), the ADetailer suite (detector downloads) and
`webui/src/ui/{rich-text,image-ref}.test.*`.

Contents: [Threat model](#threat-model) · [Controls](#controls) ·
[Running exposed](#running-exposed) · [Residual risks](#residual-risks) ·
[Audit record](#audit-record-2026-10) · [Dependency advisories](#dependency-advisories)

## Threat model

**Trust model.** The WebUI is single-tenant. Every logged-in user is an
administrator: they can generate, change settings (including output
folders), manage the queue and, where extension access is allowed, install
extensions — which is running arbitrary code. There are no roles. Without
`--auth` the server trusts everyone who can reach it, as the Gradio WebUI
did.

**What is protected:** the machine (no code execution, file reads or writes
outside what the UI offers), the user's files (outputs, uploads, models), the
login secrets, and the user's browser session.

| adversary | can | defended by |
|---|---|---|
| a web page the user visits | make the user's browser send requests to the WebUI (with its cookie), re-point a domain at `127.0.0.1` (DNS rebinding), embed the UI in a frame | cross-site write and WebSocket checks (Origin / fetch metadata), cross-site read checks, rebound-`Host` refusal, `SameSite=Lax` cookies, CSP `frame-ancestors` |
| a peer on the network (exposed server) | send any request, guess passwords | `--auth` login on everything, throttled failures, `--api-auth` for `/sdapi`, extension access off by default when exposed |
| untrusted content | images (uploads, URLs, PNG text chunks), infotext, network metadata, extension index JSON, booru API responses, extension HTML/Markdown | content sniffing, size/pixel caps, opaque file ids, no `innerHTML` in the SPA, URL scheme allow-lists, CSP, no pickle loading of uploads |
| an API client | everything a user can, through `/api/v2` and `/sdapi/v1` | the same logins; restricted settings stay out of `/sdapi`; URL fetching limited to public addresses |

**Out of scope:** extensions the administrator installed (they run with the
WebUI's privileges), other OS users with access to the data folder, and
attackers who already have a login (they are administrators by design).

## Controls

### Authentication (`remi/server/auth.py`)

- `--auth user:pass[,…]` / `--auth-file`: every request to `/api/v2` (REST
  and WebSocket), `/internal`, files and thumbnails, the SPA shell and the API
  docs (`/docs`, `/docs/*`, `/redoc`, `/openapi.json`) needs the session
  cookie or HTTP Basic credentials. Public: `/api/v2/auth/*`,
  `/internal/ping`, `/favicon.svg`, `/assets/*` (the SPA's build files,
  public so the shell can load and show its sign-in form: only existing
  build files; anything else there is a 404, never the SPA shell).
- Signing in: the server renders no login page. A browser page load
  without a login (the SPA's routes and the API docs) gets the SPA shell
  with status 401 and `Cache-Control: no-store` (no data: the built
  `index.html` plus the root path and `--theme`); the SPA asks
  `GET /api/v2/auth/status` and shows its own form (`webui/src/features/auth`),
  translated from its catalogs, which posts to `/api/v2/auth/login`. The
  status also names where the source code is (AGPL-3.0 section 13, the form
  links it; licensing.md): before a login Remi offers its repository's
  branch, never the commit it found, which would tell anyone who reaches
  the server which fixes it lacks (`--source-url` is offered as given, and
  must not carry a user name or password). A 401
  while the app is open shows the form over it. Without a built SPA (or
  under `--nowebui`) the 401 offers HTTP Basic (`WWW-Authenticate`).
- Session cookie `remi_session`: `user.expiry.HMAC` signed with a
  per-installation secret (`secret.key`, mode 0600) and the user's password;
  30 days; `HttpOnly`, `SameSite=Lax`, `Secure` with `--tls-*` or an HTTPS
  request; scoped to `--subpath`. Logout revokes it server-side (denylist
  until expiry) and closes the WebSockets opened with it (close 1008); a
  password change invalidates all sessions.
- Brute force: ten failed logins per client per minute — through the login
  form **or HTTP Basic** — and the client gets 429 for a minute, even with the
  right password. Password comparisons are constant-time; unknown users cost
  the same as wrong passwords; credentials over 1024 characters are refused
  unchecked.
- `/sdapi/v1` (core and extension routes under it): `--api-auth` Basic
  credentials when set (throttled the same way); without `--api-auth`, the
  WebUI login (cookie or Basic) on any bind — a loopback-only server is
  still reached through a reverse proxy or tunnel on the same machine. The
  explicit opt-out `--api-no-auth-on-localhost` restores the A1111
  behaviour for local tools: a loopback peer that came through no proxy (no
  `X-Forwarded-For`/`-Host`, `X-Real-IP` or `Forwarded` header) needs no
  login for `/sdapi` and extension API routes; it never applies with
  `--api-auth`, nor to `/api/v2`, files, pages or the docs.
- Docs without `--auth` but with `--api-auth` need the API credentials.
- Extension routes outside these prefixes (`/controlnet/*`, `/adetailer/*`):
  `modules/extension_route_guard.py` — cross-site checks always; a login
  whenever `--auth` / `--auth-file` is set, on any bind (session, Basic, or
  `--api-auth` credentials; local clients exempt only with
  `--api-no-auth-on-localhost`); under `--nowebui`, `--api-auth` whenever it
  is set. Refusals answer JSON `{"detail": …}` like the API's, a 401 with
  `WWW-Authenticate: Basic`, which these routes accept.

### Browser-based attacks (`modules/extension_route_guard.py`, `remi/server/auth.py`)

- **CSRF / cross-site WebSocket:** `POST`/`PUT`/`PATCH`/`DELETE` and
  WebSocket handshakes whose `Origin` is not this site (scheme, host and
  port) or an allowed `--cors-allow-origins(-regex)` origin are refused (403 /
  close 1008); `Origin: null` always is; without an `Origin`,
  `Sec-Fetch-Site: cross-site` is refused. API clients (no `Origin`) pass.
  A wildcard (`*`, or a regex that admits every origin, such as `.*`)
  allows no write: only origins listed exactly, or matched by a narrower
  regex, do.
- **CORS credentials:** allowed origins get `Access-Control-Allow-Credentials`
  and may read the answers with the user's cookie. A wildcard never does:
  with one the server sends no `Allow-Credentials` at all (to the exact
  origins next to it neither; `remi/server/app.py`), so every site reads only
  what needs no login. Starlette, told to allow credentials, answered a
  wildcard request that carried a cookie with the requesting origin and
  `Allow-Credentials: true`.
- **Cross-site reads:** a `GET`/`HEAD` that fetch metadata marks as a
  cross-site or same-site *subresource* request (`<img>`, `fetch`, frames) is
  refused unless its origin is allowed (a wildcard's included: its answers
  carry no credentials); top-level navigations (links, bookmarks) pass. This stops blind requests such as
  `<img src="http://127.0.0.1:7860/api/v2/extensions/index?url=…">`.
- **DNS rebinding:** a loopback-only server (default bind, loopback
  `--server-name`, no `--ngrok`) refuses a `Host` that is not a loopback name
  or the host of an allowed CORS origin (403 "Unknown host refused"); a
  wildcard names no host.
- **Headers:** every response carries `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: same-origin` and `X-Robots-Tag: noindex, nofollow` (an
  exposed WebUI is a private tool, not a site to index). The SPA shell and the API docs pages carry a CSP:
  scripts only from the server plus the page's own inline scripts by SHA-256
  hash (no `unsafe-inline`/`unsafe-eval` scripts), no plugins, `base-uri` /
  `form-action` `'self'`, `frame-ancestors 'self'` plus the literal
  `--cors-allow-origins` (clickjacking). `connect-src` is `'self'` (the
  API and its WebSocket; CSP Level 3 browsers, which the SPA's ES2023 build
  requires anyway, match `ws:`/`wss:` to the page's own origin) plus
  `https://raw.githubusercontent.com` for the license texts of Settings →
  Licenses, so an injected script could not send data elsewhere. `img-src`
  is `'self' data: blob:` plus the hosts the user lists in Settings → System
  → "Remote image hosts" (`remi_image_hosts`; only `http(s)://host[:port]`
  and `https://*.host` tokens are taken, so the setting cannot add other
  sources or directives); remote images in extension help or outputs are
  blocked otherwise, so injected markup cannot load tracking pixels.

### Files and uploads (`remi/files.py`, `remi/server/routes/files.py`)

- Files are served only by opaque id (`/api/v2/files/{id}`,
  `/api/v2/thumbs/{id}`; ids match `^[A-Za-z0-9_-]{8,64}$`); the browser never
  sends or receives a filesystem path. Non-media files are sent as
  attachments (`application/octet-stream`); SVG is never accepted.
- Uploads: 64 MB by default (`remi_upload_max_mb`), enforced while streaming
  (also for chunked bodies), one file per form; the type comes from the
  content (PNG, JPEG, GIF, WebP, BMP, TIFF, JXL, AVIF/HEIC, MP4/MOV/WebM/AVI,
  safetensors headers, UTF-8 text by extension), never from the name or
  declared type. Images are capped at 179 MP and 2000 frames from the header
  before any decode, independently of `Image.MAX_IMAGE_PIXELS` (which the
  X/Y/Z plot lifts). Stored as `uploads/<sha256><ext>`; the display name is
  cleaned (last path component, no control characters, ≤ 200 characters).
- `/sdapi` images (base64 or URL) get the same pixel cap before decoding.
- So do the images of an imported Agent Scheduler queue
  (`remi/jobs/importers.py`): its restricted unpickler checks a pickled
  image's declared size before Pillow allocates it and never calls an image
  class (`PngImageFile(path)` would open a file on the server); image dicts
  and data URLs are checked before `frombytes` / `load`. Numpy arrays and
  scalars are taken only in the form numpy pickles them, with data that
  fits the declared shape, never `ndarray(shape)` or an array of objects,
  and their dtype is made again from its description (a pickled dtype can
  claim to hold objects, and numpy would take raw bytes for pointers).
- Upload errors are fixed messages ("not a readable image"): Pillow's own
  message names the incoming file in the data folder, so it is only logged.
- Every other request body (JSON jobs, `/sdapi` base64 images, extension
  routes) is capped at twice `remi_upload_max_mb` (at least 16 MB),
  enforced while it streams in, chunked bodies included
  (`remi/server/limits.py`): 413 past it, without buffering more. Paths
  that answer without a login (`PUBLIC_PATHS`: the login form, the ping)
  take at most 64 KB: a caller nobody has vouched for cannot make the
  server buffer and scan megabytes.
- Unsafe deserialization: uploads can only be images, video, text or
  safetensors (a JSON header); nothing reachable from an upload or URL calls
  `pickle`/`torch.load`. Model files are only loaded from the model folders,
  with one exception: the Prompt Upsampler's `local_model_path`
  (`POST /api/v2/prompt-tools/upsample` and a pipeline's upsample stage)
  loads a `.gguf` file (llama-cpp) or a model folder (transformers,
  `local_files_only`, no remote code) from any path the server can read;
  a crafted folder can still run code through transformers 4.x
  (GHSA-fgcw-684q-jj6r, see [Dependency advisories](#dependency-advisories)),
  so name only folders from a source you trust, as with any model file.
  `--hide-ui-dir-config` refuses it (see below).

### Paths and local actions

- Static SPA files are resolved inside `webui/dist` only; locale, preset,
  page, network and config-state names are looked up in known lists, never
  joined into paths. Network metadata and previews are written next to an
  indexed model file only.
- Opening folders (`/outputs/open-folder`, `/networks/…/open-folder`,
  `/gallery/{id}/reveal`) is refused with `--hide-ui-dir-config` and for any
  client that is not a loopback peer without proxy headers. On macOS a folder
  that is a bundle (`Some.app`) is revealed in Finder, never launched.
- Path settings are refused with `--hide-ui-dir-config` (`restricted_opts`),
  and `--freeze-settings*` freeze settings for `/api/v2` and `/sdapi` alike
  (the prompt tools' translation key, kept outside `config.json`, freezes with
  `--freeze-settings` or a frozen `prompt_tools` section).
  So are job inputs that name a path on the server (img2img batch from a
  directory, Extras batch from a directory and its video clip): the form
  hides them and `check_core` (`remi/core/forms.py`) refuses a request that
  still sends one, whichever way the job arrives. Without the flag, the
  Extras video clip must still be an existing regular file: `av.open` and
  `ffmpeg -i` read a URL or a protocol prefix (`http:`, `concat:`,
  `subfile,`…) as one and would fetch it from the server's network, so such
  a value is refused (a Windows drive letter is a path).
- `--hide-ui-dir-config` also refuses the Prompt Upsampler's
  `local_model_path` (403 from `POST /api/v2/prompt-tools/upsample`; a
  pipeline's upsample stage fails), stops an empty infotext paste
  (`POST /api/v2/infotext/parse`) from reading the last generation's
  `params.txt`, as `--no-prompt-history` does (only that flag stops writing
  it), and leaves the file paths out of the queue import sources and
  reports (`path: null`, `/api/v2/queue/import`).

### Extensions (`remi/extensions_admin.py`)

- Installing, updating, uninstalling, restoring and the index download are
  refused (403) when the server is reachable from other machines, unless
  `--enable-insecure-extension-access`.
- Install input: URL `https://`, `http://`, `ssh://`, `git://` or
  `user@host:path`, no control characters, a host or user that cannot be read
  as an option (`ssh://-oProxyCommand=…`); branch names without a leading `-`
  or `/`, `..`, `//` or a `.lock`/`/`/`.` ending; folder names
  `[A-Za-z0-9][A-Za-z0-9._-]{0,99}`, no Windows device names. Git runs without
  a shell, with `--` before the URL, no terminal or credential prompts, a
  timeout, and `GIT_ALLOW_PROTOCOL=https:http:ssh:git` for the clone and its
  submodules (no `file://` or `ext::` helpers). Uninstall only deletes a
  direct child of `extensions/`.
- The index is fetched only over http(s), at most 16 MB, entries whose URL
  fails the same validation are dropped. An index URL the client names
  (other than the default or `WEBUI_EXTENSIONS_INDEX`) must resolve to public
  internet addresses only, and so must every redirect hop (422 otherwise),
  so the field cannot be used to probe the server's own network.

### Outbound network calls

Nothing contacts the network unless the user configured or asked for it:

| call | when |
|---|---|
| extension index (`raw.githubusercontent.com` or `WEBUI_EXTENSIONS_INDEX`) | opening Extensions → Available; another index URL typed there must be a public address (each redirect hop checked) |
| `git fetch` / `clone` | checking for updates, installing, updating |
| prompt translation | only with a network provider and its URL/key set (`prompt_tools_translation_*`); the key is write-only through the API |
| prompt writing ("Write a prompt", `POST /api/v2/prompt-tools/generate`) | only when the user sends a description and the translation provider is the OpenAI-compatible one with its URL and model set; the description (and the current prompt, when changing it) goes to that provider's `/chat/completions` with the same write-only key |
| booru grabber (Prompt Workshop) | on a grab; https only, the booru's own image hosts, public addresses checked per redirect hop, 64 MB cap |
| model downloads (upscalers, face restorers, preprocessors, ADetailer, TAESD) | the first time the user runs a feature whose model is missing; fixed upstream URLs. ADetailer lists its 10 stock detectors from the start and downloads one when a unit first runs with it (never at startup or on Refresh; `--ad-no-huggingface`: never). A test starts the whole WebUI with an empty data folder and fails on any connection to a non-local host |
| `/sdapi` image URLs | an API request names one; off with `api_enable_requests`; with `api_forbid_local_requests` (default) every address of every redirect hop must be public; 64 MB cap. Both settings are `restrict_api`: `/sdapi/v1/options` cannot change them |
| license texts | opening an entry in Settings → Licenses (fetched by the browser from `raw.githubusercontent.com`, the only outside host the CSP lets the page connect to, without cookies or referrer) |
| remote images | only from hosts listed in "Remote image hosts" (`remi_image_hosts`, empty by default) |

The API docs (`/docs` Swagger UI, `/redoc` ReDoc, `remi/server/api_docs.py`)
are not FastAPI's default pages, which load floating CDN versions
(`cdn.jsdelivr.net/npm/swagger-ui-dist@5`, `redoc@2`), Google Fonts and a
`fastapi.tiangolo.com` favicon into a page on the app's origin. The SPA build
vendors pinned copies (`swagger-ui-dist` and `redoc` in `webui/package.json`,
`apiDocs` in `webui/vite.config.ts`, which writes them under
`assets/<package>-<version>/` with their licenses and fails when a viewer
still names a CDN host; ReDoc's attribution logo from `cdn.redoc.ly` is
replaced). The pages carry the same CSP as the SPA shell (scripts from the
server and the page's inline script by hash, `connect-src 'self'`), Swagger
UI's validator badge (`validator.swagger.io`) is off, and the packages'
install scripts are disabled (`webui/pnpm-workspace.yaml`: `@scarf/scarf`
would report the install). `tests/test_remi_api_docs.py` checks the pages;
a browser load of both against a built server makes no outside request and
reports no CSP violation.

### XSS (`webui/src/ui/rich-text.tsx`, `webui/src/ui/image-ref.ts`)

- The SPA never uses `innerHTML`/`dangerouslySetInnerHTML`; React escapes all
  text (infotext, file names, network metadata, extension strings).
- Markdown nodes are rendered by a small parser into React elements; raw
  HTML stays text. HTML nodes/outputs are parsed inert (DOMParser) and
  rebuilt from an allow-list of tags and attributes; scripts, styles,
  frames, forms, SVG and event handlers are dropped.
- Link and image URLs pass `safeUrl` (http, https, mailto or relative, read
  with the browser's URL parser, so `java\tscript:` and entity tricks fail);
  output images pass `safeImageUrl` (http(s), blob, `data:image/*`, same-site
  paths). The CSP is the second line of defence.

### Error responses (`remi/server/app.py`)

An unhandled exception answers `500 {"error": "Internal Server Error",
"detail": "Internal server error (id …); …", "id": "…"}`: no exception text,
no request or response body (a `ResponseValidationError` carries the whole
response), no paths. The exception and traceback are logged with the same
id. An `HTTPException` keeps its status, and a 4xx keeps its `detail`, which is
meant for the client. A status on any other exception is ignored: an SDK's
`AuthenticationError` (`status_code` 401) that an extension lets escape is a
500, not a 401 that would show the sign-in form.

A request that fails its model is a 422, logged at debug level without a
traceback, so no client (the public login form included) can flood the log.
That holds for bodies with `1e999` or `NaN` too: the error responses encode
non-finite numbers as text, and `/api/v2` models refuse them as values.
`POST /sdapi/v1/options` with a setting that does not exist is a 422 that
sets nothing (it was a `KeyError`, a 500). So is a changed value its
setting's widget refuses (the wrong kind, a choice not offered): it reached
upstream code that reads the setting as it is stored and failed there until
the next start.

Text UTF-8 cannot hold (an unpaired surrogate, which JSON can escape) is
refused with a 422 in any JSON body, `/api/v2` and `/sdapi/v1` alike
(`refuse_broken_text`, a dependency of the whole app), and per setting in
`/options`, before it reaches a file or SQLite: written, it failed half-way
through the write (a truncated `config.json` the server could not start
from). Responses write such text with JSON escapes rather than fail (an
extension can still store it).

### Secrets in logs

`modules/redact.py`: the launch line, `sysinfo` (command line,
`COMMANDLINE_ARGS`, settings with key/token/secret/password-like names,
extension remotes) and `/sdapi/v1/cmd-flags` hide `--auth`, `--api-auth`,
`--ngrok`, `--ngrok-options` values (also as `--flag=value` and argparse
abbreviations) and credentials inside URLs. A failed `--ngrok` connection
names the token by its last 4 characters only (`modules/ngrok.py`). Logins
are never logged; the translation API key is stored write-only.

## Running exposed

The default (no `--listen`) only accepts connections from this machine.
To let other machines in:

1. **Always set a login:** `--auth user:password` or `--auth-file logins.txt`
   (one or more `user:pass` per line). Use long random passwords; the server
   prints a warning at startup when it is reachable without one.
2. **Use TLS:** `--tls-keyfile key.pem --tls-certfile cert.pem`, or put the
   WebUI behind an HTTPS reverse proxy (bind it to loopback and let the proxy
   forward; the session cookie becomes `Secure` when the proxy reports
   HTTPS). Without TLS, passwords and cookies cross the network in clear text.
   The WebUI sends no `Strict-Transport-Security` (HSTS) header, not even
   with `--tls-*`, so a browser sent to `http://` (a typed name, an old link,
   an attacker on the network) does not insist on HTTPS. Let the HTTPS proxy
   add it, e.g. nginx
   `add_header Strict-Transport-Security "max-age=31536000" always;`.
3. **API:** `--api` exposes `/sdapi/v1`; set `--api-auth` for it (otherwise
   it uses the WebUI login). Scripts can send Basic credentials. Do not use
   `--api-no-auth-on-localhost` when other people can run programs on the
   machine or when a proxy that adds no forwarding headers fronts it (the
   server prints a startup warning while it is on).
4. **Keep extension access off** (the default when exposed). Only add
   `--enable-insecure-extension-access` if every user may run code on the
   machine.
5. **Hide paths** with `--hide-ui-dir-config` when users should not change
   output folders or open folders on the server, and consider
   `--freeze-settings` / `--freeze-settings-in-sections`.
6. **CORS:** list exact origins in `--cors-allow-origins` (or a narrow
   `--cors-allow-origins-regex`) only for tools you trust: allowed origins
   may send requests with the user's cookie and read the answers. `*`, or a
   regex that admits every origin (`.*`), lets every site read what the
   server answers without a login, never with one, and change nothing;
   without `--auth` that is everything, so never use it on an exposed server.
7. **Reverse proxies:** forward `Host` (or `X-Forwarded-Host`) and the
   WebSocket upgrade for `/api/v2/ws`; with a path prefix use `--subpath`.
   Login throttling counts per peer address: `X-Forwarded-For` and
   `X-Forwarded-Proto` are ignored (a client of a same-machine tunnel could
   otherwise name a new address per guess), so behind a proxy all clients
   share one counter (a failed-login flood locks everyone out for a minute,
   which errs on the safe side). Name your proxies with
   `--forwarded-allow-ips` (e.g. `127.0.0.1` for one on this machine) to
   count each client on its own; the proxy must then append (or set) the
   address it saw in `X-Forwarded-For` (nginx with
   `$proxy_add_x_forwarded_for`, Caddy by default), as the rightmost entry
   that is not a named proxy is taken.
8. `--ngrok` tunnels count as exposed (HTTPS is provided by ngrok).

## Residual risks

- **Exposed without `--auth`:** anyone who can reach the server controls it,
  and a site that a user on the same network visits can reach it through DNS
  rebinding (the `Host` check only protects loopback-only servers). Use a
  login.
- **Same-machine tunnels without forwarding headers** (`ssh -L`, plain TCP
  forwarders) look like local clients to the folder actions and, with
  `--api-no-auth-on-localhost`, to the API; use `--hide-ui-dir-config` and
  leave that flag off when sharing that way.
- **DNS answers that change between the address check and the connection**
  are not caught for `/sdapi` image URLs (turn `api_enable_requests` off on
  servers that must not reach their network), nor for a client-named
  extension index URL (only reachable where extension access is allowed).
- **Slow clients:** uvicorn has no timeout for a request's headers (only
  for an idle kept-alive connection), so a client can hold connections open
  by sending them slowly. On a server others can reach, put a reverse proxy
  with header and body timeouts in front.
- **Request bodies** are buffered up to their cap (twice
  `remi_upload_max_mb`): each concurrent request of a client allowed to call
  the API can hold that much memory while it is parsed.
- **Model downloads** come from fixed upstream URLs and trust those hosts and
  TLS; downloaded `.pth`/`.pt` files are loaded by their libraries
  (`torch.load` defaults to `weights_only=True` in the bundled PyTorch).
- **Installed extensions** are trusted code: their routes are guarded, but
  what they do inside is theirs.

## Audit record (2026-10)

Findings of the wave-5 audit and their fixes (severity for an exposed
server with `--auth`, or a local server visited by a hostile page).

| # | finding | severity | fix |
|---|---|---|---|
| 1 | `/sdapi/v1/options` called `set_config` without `is_api`, so API clients could turn off `api_forbid_local_requests` / `api_enable_requests` (then SSRF) | high | `set_config(req, is_api=True)` |
| 2 | `/sdapi` image URLs: redirects followed blindly after checking only the first host (SSRF to internal hosts), IPv4-only resolution, no size limit | high | manual redirects with a check per hop, `getaddrinfo` (IPv4+IPv6, mapped addresses), 64 MB cap |
| 3 | `/sdapi` and extension `/sdapi/*` routes were open on an exposed server with `--auth` but no `--api-auth`; extension `/sdapi/*` routes ignored `--api-auth` on loopback | high | `sdapi_refusal`: `--api-auth`, else the WebUI login when exposed |
| 4 | HTTP Basic credentials were checked on every request without throttling (unlimited password guessing) | medium | per-client throttle shared with the login form; `--api-auth` throttled too |
| 5 | `GET /sdapi/v1/cmd-flags`, the launch log line and `sysinfo` (`--auth=…` form, `--ngrok` token, `COMMANDLINE_ARGS`, URL credentials) leaked secrets | medium | `modules/redact.py` |
| 6 | Cross-site `GET`s (e.g. `<img src=…/extensions/index?url=http://intranet/>`) ran with no check: blind SSRF/CSRF through side-effecting reads | medium | fetch-metadata check for subresource reads |
| 7 | `safeUrl` let `java\tscript:` (and entity-encoded tabs in HTML nodes) through; output image `url`s were used as links unchecked | medium | URL-parser-based allow-list; `safeImageUrl` |
| 8 | Upload decompression-bomb protection depended on `Image.MAX_IMAGE_PIXELS`, which the X/Y/Z plot sets to `None` for the whole process | medium | explicit 179 MP / 2000-frame cap from the header; same for `/sdapi` images |
| 9 | No CSP or framing protection (clickjacking of the extension installer, no XSS backstop) | medium | CSP with inline-script hashes, `frame-ancestors`; `nosniff` and `Referrer-Policy` everywhere |
| 10 | Open-folder endpoints could be triggered by remote clients; on macOS a "folder" that is an `.app` bundle was launched | low | loopback-only like reveal; bundles revealed |
| 11 | Extension install: `ssh://-o…` style hosts and `file`/`ext` transports in submodules were not excluded; `..` branches and Windows device names accepted | low | `valid_git_url` / `valid_branch` / `valid_dirname`, `GIT_ALLOW_PROTOCOL` |
| 12 | `/docs/oauth2-redirect` was outside the docs login; docs open under `--api-auth` without `--auth` | low | docs prefix check, `--api-auth` for docs |
| 13 | Unbounded login field sizes; the login page interpolated `--subpath` into a script unescaped | low | 1024-character cap; JSON-escaped root (the server-side login page was later removed: the SPA signs in) |
| 14 | No warning when the server is exposed without a login, with `--cors-allow-origins=*`, insecure extension access or without TLS | low | startup warnings (`remi.server.launch.exposure_warnings`) |
| 15 | With `--auth` and no `--api-auth`, `/sdapi` and extension API routes (`/controlnet/*`, `/adetailer/*`) were open on a loopback-only server, which a reverse proxy on the same machine makes reachable from anywhere | high | the WebUI login on any bind; explicit `--api-no-auth-on-localhost` opt-out for local tools |
| 16 | The error handler returned the exception text and its `body` (`GET /sdapi/v1/cmd-flags` echoed the whole flag set, with paths, in a `ResponseValidationError` because `--port` is an int with a `None` default) | medium | generic 500 with a correlation id, details only in the log; `FlagsModel` types from the parser's `type=`; the unused A1111 `api_middleware` (same leak, one call away from being re-enabled) removed |
| 17 | CSP allowed `connect-src https: ws: wss:` and `img-src *`: an XSS could exfiltrate to any host, markup could load tracking pixels | medium | `connect-src 'self'` + the license host; `img-src 'self' data: blob:` + an opt-in host list |
| 18 | `GET /api/v2/extensions/index?url=` fetched any client-named URL, including loopback and private addresses (port/service probe through the error text) | low | client-named index URLs and their redirects must be public addresses |
| 19 | ADetailer downloaded its 10 detector models (about 240 MB from Hugging Face, GitHub and Google Storage) at startup, with no user action | medium | listed at startup, downloaded when a unit first runs with one; the startup test runs without `--ad-no-huggingface` |
| 20 | JSON bodies (jobs, `/sdapi` base64 images) had no size cap | medium | twice the upload limit, enforced while streaming (`remi/server/limits.py`) |
| 21 | `{"username": 1e999}` to the public login (and any body route) answered 500 with a logged traceback per request: the validation error could not be encoded | low | 422 through a non-finite-safe handler, debug log; `/api/v2` models refuse non-finite numbers |
| 22 | `/assets/` and `/assets/<missing>` served the SPA shell without a login | low | 404 for anything under `/assets` that is not a build file |
| 23 | An unreadable upload's 415 named the incoming file's path in the data folder (also the network preview, PNG Info and reveal errors could carry OS error text) | low | fixed messages; the details are logged |
| 24 | `/docs` and `/redoc` were FastAPI's defaults: Swagger UI / ReDoc scripts from `cdn.jsdelivr.net` (floating `@5` / `@2`, no SRI), Google Fonts and a `fastapi.tiangolo.com` favicon on every visit, third-party code on the app origin with the session, no CSP | medium | own pages with the build's pinned copies and the shell's CSP (`remi/server/api_docs.py`) |
| 25 | uvicorn trusted `X-Forwarded-For` from loopback peers by default, so a client reaching a loopback-bound server through a same-machine tunnel (or any local program) could rotate the header and guess passwords without ever hitting the throttle | medium | proxy headers only from `--forwarded-allow-ips`; the `Secure` cookie flag still follows a TLS proxy's `X-Forwarded-Proto` (`remi.server.auth.reached_over_https`); real-server regression tests in `tests/test_remi_server_auth.py` |

Checked and found sound: session signing and revocation, cookie flags,
login timing, Origin/rebinding checks for writes and WebSockets, file serving
by id, upload streaming limits and content sniffing, SPA static path
resolution, network write paths, config-state ids, the booru grabber's URL
checks, the translation key handling, and the absence of `innerHTML` in the
SPA.

## Audit record (2026-11)

| # | finding | severity | fix |
|---|---|---|---|
| 1 | The SPA shell under another name for the same file (`/index.html/`, `/Index.html` on a case-insensitive volume, a hard link) was served as a plain file, without its CSP and `frame-ancestors`: any site could frame it (clickjacking the extension installer, deletes, settings) | high | the shell is recognised by file identity and never served as a plain file |
| 2 | Text with an unpaired surrogate in a setting, a style, a recipe or a pipeline failed half-way through its write: `config.json` was left truncated and the next start failed | high | refused with a 422 before any write (see Error responses) |
| 3 | Infotext parsing (`parse_infotext_fields`, LoRA tags) backtracked quadratically: a crafted 40,000-character infotext held the process for seconds per request (the compare request had no size cap; scans stored any length) | medium | linear expressions, parsing cut at `MAX_INFOTEXT_CHARS`, stored infotext truncated, `max_length` on the compare request; the SPA's parsers likewise |
| 4 | An unreadable `Origin` (`http://[`) or `--cors-allow-origins` entry answered 500 with a logged traceback before the login check | low | read as a foreign origin |
| 5 | A session cookie whose signature held non-ASCII characters answered 500 with a logged traceback before the login check, on every route | low | refused like a wrong signature |
| 6 | `GET /api/v2/auth/status` checked HTTP Basic credentials without the login throttle (a public path) and answered who they log in as: unlimited password guessing again (2026-10 #4) | medium | public paths read the session cookie only; `/auth/status` checks Basic itself, counting failures and telling a throttled client nothing |
| 7 | A request to a path open without a login (`/api/v2/auth/*`) could carry a body up to the full cap (128 MB by default), buffered and scanned before any login check (a 60 MB body took about 180 MB of memory) | low | 64 KB on those paths, enforced while the body streams in |
| 8 | `POST /sdapi/v1/options` with text holding an unpaired surrogate stored it and saved the settings: the save stopped half-way and left `config.json` cut short (#2 for the A1111 API), every setting lost at the next start | high | `/sdapi/v1` JSON bodies are checked as `/api/v2`'s are (422, nothing stored) |
| 9 | A failed `--ngrok` connection printed the whole authtoken to the console, which launchers and services often log | low | the token is named by its last 4 characters; the authtoken (also `NGROK_AUTHTOKEN`), the basic-auth credentials and any `--ngrok-options` value under a secret-like name (`auth`, `secret`, `token`, `password`, `key`) are replaced by `***` in the printed reason, unless 4 characters or shorter (replacing so short a text would garble the message and still show where it was) |
| 10 | A setting value nested about a thousand levels deep, posted to `/sdapi/v1/options` (a model pick too), `PATCH /api/v2/options` or a generation's `override_settings`, passed the checks and stopped `Options.save`'s recursive `json.dump` half way: `config.json` cut short (#2 again), every setting lost at the next start | high | a list or object nested deeper than 64 levels (`MAX_NESTING`) is refused on those routes before anything is set (422); `Options.save` writes the file whole or not at all (`modules/options.py`), so a value that reaches the settings another way can no longer cut it short |

## Dependency advisories

The advisories on the pinned Python packages, each checked against the code
path it needs (advisory data from OSV). `transformers` and `diffusers` are the
model runtime shared with Forge Neo, which pins the same versions: a pin moves
with an upstream sync unless an advisory is reachable here and a patch release
of the same line fixes it. Look again at each upstream sync
([upstream-absorption.md](upstream-absorption.md)), or sooner when Remi starts
using a path named below.

### 2026-11-26: Starlette 1.3.1 and FastAPI 0.136.3

Every request goes through Starlette, so its advisories are fixed whether or
not Remi's routes reach them: `starlette==1.3.1` is pinned in
`requirements.txt` itself (FastAPI only sets a floor an older Starlette still
meets) with `fastapi==0.136.3`, the last release before FastAPI 0.137 stopped
listing included routes in `app.routes` (extensions that walk the routes keep
working). `tests/test_dependencies.py` keeps both pins at or above the fixes.

| advisory (Starlette 0.50.0) | fixed in | in Remi |
|---|---|---|
| GHSA-82w8-qh3p-5jfq (CVE-2026-54283, high): `request.form()` ignores its field and size limits for `application/x-www-form-urlencoded` (memory and CPU denial of service) | 1.3.1 | no route reads `request.form()`, by itself or through a `Form` or `File` parameter (uploads parse multipart themselves, with caps) and request bodies are capped (`remi/server/limits.py`); an extension's form route would have been exposed |
| GHSA-wqp7-x3pw-xc5r (CVE-2026-48818, high): `StaticFiles` on Windows follows UNC paths (SSRF, NTLM credential leak) | 1.1.0 | no `StaticFiles` mount: the SPA's files are resolved inside `webui/dist` |
| GHSA-86qp-5c8j-p5mr (CVE-2026-48710, moderate): a crafted `Host` header poisons `request.url.path` | 1.0.1 | the login, cross-site and route checks read the ASGI scope's path (`route_path`); `request.url` only appears in log lines |
| GHSA-x746-7m8f-x49c (CVE-2026-48817, moderate): `HTTPEndpoint` dispatches any HTTP method name to an attribute | 1.1.0 | no `HTTPEndpoint` |
| GHSA-jp82-jpqv-5vv3 (CVE-2026-54282, low): the request path poisons `request.url.hostname` | 1.3.0 | the `Host` check reads the header (`modules/extension_route_guard.py`) |

FastAPI 0.132 stopped reading a body without a `Content-Type` as JSON. The app
turns that off (`strict_content_type=False` in `remi/server/app.py`): A1111
API clients post JSON without one, as Forge Neo still accepts. A cross-site
page can send such a body too (a `Blob`), and the cross-site check refuses it
by its `Origin` and `Sec-Fetch-Site`, as before
(`tests/test_remi_security.py`).

### 2026-11-26: transformers 4.57.6 and diffusers 0.37.1 kept

Every fix is in a new line (transformers 5.x, diffusers 0.38; 4.57.6 and
0.37.1 are the last of theirs), whose model-loading changes need the GPU
regression runs. One path is open to a client, but only through a crafted
model folder already on the server's disk, which no route lets a client
write (GHSA-fgcw-684q-jj6r below).

| advisory | fixed in | why it does not apply |
|---|---|---|
| transformers GHSA-29pf-2h5f-8g72 (CVE-2026-4372, high): a `config.json` that names a Hub kernel as `_attn_implementation_internal` makes `from_pretrained` download and run it | 5.3.0 | the setting is read (a model folder's config reaches it), but a Hub kernel loads only through the `kernels` package, which is no requirement and not installed: the load stops with an `ImportError` before anything is downloaded. Revisit if `kernels` is ever installed |
| transformers GHSA-fgcw-684q-jj6r (CVE-2026-5241, high): a LightGlue `config.json` turns `trust_remote_code` on for the config it nests | 5.5.0 | a client names the folder: the Prompt Upsampler's `local_model_path` is a field of `POST /api/v2/prompt-tools/upsample` and of a pipeline's upsample stage, and `remi/prompt_tools/pup.py` loads any folder the server can read. A crafted LightGlue `config.json` there runs code from another folder on the disk (checked offline with `AutoTokenizer.from_pretrained(..., local_files_only=True)`), or from a Hub repository it names, since the nested load does not pass `local_files_only` (read in `configuration_lightglue.py`, not run). No route lets a client write such a file: uploads are stored as their SHA-256 with the extension their content shows, a save copies only outputs under the names the server gave them, settings take only known keys, and installing an extension (which writes anything, and is code execution already) is refused off loopback without `--enable-insecure-extension-access`. Refused under `--hide-ui-dir-config`; the checkpoint loader reads configs only from the repository's `backend/huggingface/`, and JoyCaption downloads one fixed repository. Use model folders from sources you trust |
| transformers GHSA-x9r9-c232-4q39 (CVE-2026-80047, high): `load_custom_generate` caches a repository's `custom_generate/generate.py` before the trust check | none yet | Remi never passes `custom_generate` or `trust_remote_code`; a generation mode that moved to a Hub repository raises before anything is fetched |
| transformers GHSA-xrqw-3rrv-vx5w (CVE-2026-9856, high): a tokenizer's or processor's `save_pretrained` writes outside its folder through chat template names | 5.10.0 | Remi never saves a tokenizer or a processor |
| transformers GHSA-69w3-r845-3855 (CVE-2026-1839): `Trainer` loads its RNG state with `torch.load` | 5.0.0 | no training; the bundled PyTorch loads with `weights_only=True` |
| transformers PYSEC-2025-217 (CVE-2025-14929): the X-CLIP checkpoint conversion script deserializes untrusted data | none in 4.x | conversion scripts are never run |
| diffusers GHSA-98h9-4798-4q5v (CVE-2026-44513, high), GHSA-7wx4-6vff-v64p (CVE-2026-45804, high), GHSA-j7w6-vpvq-j3gm (CVE-2026-44827): `DiffusionPipeline.from_pretrained` runs a repository's or folder's custom pipeline code without `trust_remote_code` | 0.38.0 | Remi builds no `DiffusionPipeline`: the loader reads the `model_index.json` of the repository's own `backend/huggingface/` folders (`DiffusionPipeline.load_config`, JSON only) and builds the model classes named there; a user's Diffusers folder (Z-Image, ERNIE, Fibo) is read as JSON and safetensors; `modules_forge/diffusers_patcher.py`, which would build a pipeline, is used nowhere |
