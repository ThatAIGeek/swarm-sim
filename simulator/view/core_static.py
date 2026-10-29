"""Serve or inline the Core System kit from Python servers, with no build step.

Generated alongside the kit by tools/export.js (copied as is).

http.server / BaseHTTPRequestHandler (hub, pipectl, uavlabel viewer):

    import core_static
    CORE = os.path.join(os.path.dirname(__file__), "static", "core")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if core_static.serve(self, CORE):      # answers /core/... and returns True
                return
            ...                                    # the existing routes

FastAPI (farsight viewer):

    from fastapi.staticfiles import StaticFiles
    app.mount("/core", StaticFiles(directory=CORE), name="core")

One HTML string (swarm-sim's console and its --export single file): put <!--__CORE__--> in <head>, then

    html = html.replace("<!--__CORE__-->", core_static.inline(CORE))
"""

import os
import urllib.parse

TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".ttf": "font/ttf",
    ".txt": "text/plain; charset=utf-8",
}


def resolve(core_dir, url_path, prefix="/core/"):
    """Map a request path under prefix to (file, content type) inside core_dir, or None.

    Refuses anything outside the kit: '..', hidden files, unknown file types."""
    if not url_path.startswith(prefix):
        return None
    rel = urllib.parse.unquote(url_path[len(prefix):].split("?", 1)[0].split("#", 1)[0])
    parts = [p for p in rel.replace("\\", "/").split("/") if p]
    if not parts or any(p in (".", "..") or p.startswith(".") for p in parts):
        return None
    root = os.path.realpath(core_dir)
    full = os.path.realpath(os.path.join(root, *parts))
    if os.path.commonpath([full, root]) != root or not os.path.isfile(full):
        return None
    ctype = TYPES.get(os.path.splitext(full)[1].lower())
    return (full, ctype) if ctype else None


def serve(handler, core_dir, prefix="/core/"):
    """For BaseHTTPRequestHandler.do_GET / do_HEAD: answer a kit request and return True, else False."""
    hit = resolve(core_dir, handler.path, prefix)
    if not hit:
        return False
    full, ctype = hit
    with open(full, "rb") as f:
        body = f.read()
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(body)
    return True


def inline(core_dir, lab=True):
    """<style> and <script> blocks carrying the whole kit, for pages served or exported as one HTML string.

    lab=True embeds Golos Text and IBM Plex Mono only (about 0.6 MB); lab=False adds Lora."""
    def read(rel):
        with open(os.path.join(core_dir, *rel.split("/")), encoding="utf-8") as f:
            return f.read()

    tokens = read("tokens-embedded-lab.css" if lab else "tokens-embedded.css")
    css = tokens + "\n" + read("components/bundle.css")
    js = read("components/bundle.js")
    return ("<style>\n" + css.replace("</style", "<\\/style") + "\n</style>\n"
            "<script>\n" + js.replace("</script", "<\\/script") + "\n</script>\n")
