import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from sonic_bridge.server import Application, make_handler


@pytest.fixture
def live_server(tmp_path):
    app = Application(tmp_path, backend="demo")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield app, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    app.pool.shutdown()


def test_api_state_static_and_csrf(live_server):
    app, base = live_server
    with urllib.request.urlopen(base + "/api/state") as response:
        state = json.load(response)
    assert state["synthetic"] and state["token"] == app.token
    with urllib.request.urlopen(base) as response:
        assert b"Find the moment" in response.read()
        assert response.headers["Content-Security-Policy"]
    req = urllib.request.Request(base + "/api/demo", data=b"{}", headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as failure:
        urllib.request.urlopen(req)
    assert failure.value.code == 403
    req.add_header("X-Sonic-Token", state["token"])
    with urllib.request.urlopen(req) as response:
        assert json.load(response)["synthetic"]


def test_rejects_index_path_traversal(live_server):
    app, _ = live_server
    with pytest.raises(ValueError):
        app.database("../secret")


def test_raw_jepa_cannot_search_and_invalid_json_is_reported(live_server):
    app, base = live_server
    app.backend = "jepa"
    for body in [[], {"query": "dog", "index": "library"}]:
        req = urllib.request.Request(base + "/api/search", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "X-Sonic-Token": app.token})
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(req)
        assert failure.value.code == 400


def test_job_cancellation_is_cooperative_and_recorded(tmp_path):
    app = Application(tmp_path, backend="demo")
    entered, release = threading.Event(), threading.Event()

    def callback(progress):
        entered.set()
        release.wait(timeout=5)
        progress("Checkpoint")
        return {"unexpected": "completion"}

    key = app.submit("experiment", callback)["job"]
    assert entered.wait(timeout=5)
    assert app.cancel(key)["cancellation_requested"]
    release.set()
    app.pool.shutdown(wait=True)
    assert app.jobs[key]["status"] == "cancelled"
