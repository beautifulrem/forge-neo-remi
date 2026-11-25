# Licensing

Remi WebUI is licensed under the [GNU Affero General Public License v3.0](../../LICENSE),
like the Forge Neo, Forge and AUTOMATIC1111 WebUIs it derives from. Code
from other projects keeps its own license and notices: Settings → Licenses
lists the projects (`html/licenses.html`), and the web interface build's
`THIRD-PARTY-NOTICES.txt` lists every library it ships with its license text
(`webui/tools/notices.mjs`).

## Source code for users of the server (AGPL-3.0 section 13)

Section 13 asks a modified program that users interact with over a network
to offer those users its source. The web interface links the source of the
server it talks to in three places:

- **the sign-in screen** (under the form, also in the form shown over the
  app when a session ended): with `--auth`, people who cannot sign in reach
  nothing of the program but this form, and they are offered the source there;
- **Settings → Licenses**, under the WebUI's own license;
- **Settings → Server and models**, next to the version.

The link comes from the server: `source_url` of `GET /api/v2/auth/status`
(a public path) and of the bootstrap's `auth` (`remi/server/source.py`).

| case | link |
|---|---|
| default, the running commit is published in the repository (a remote-tracking ref of a remote pointing at `github.com/beautifulrem/forge-neo-remi` contains it) | `https://github.com/beautifulrem/forge-neo-remi/tree/<commit>` |
| default, the commit is unknown (no git checkout, such as a Docker image) or not published there (a local commit) | `https://github.com/beautifulrem/forge-neo-remi/tree/remi` |
| `--source-url URL` (or `REMI_SOURCE_URL` when the flag is absent) | `URL`, as given; an address that is not `http(s)` is ignored with a warning in the console |
| a client that has not signed in (with `--auth`) | the default's branch link, or `URL`: never the running commit, which would tell anyone who reaches the server which fixes it lacks |

**Running a modified copy:** publish the source of what you run (your
changes included) and start the server with `--source-url` pointing at it,
for example `--source-url https://git.example.com/me/remi/src/commit/<commit>`.
Remi cannot tell where you publish which version, so it links that URL as
it is.
