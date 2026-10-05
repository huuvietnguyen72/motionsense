from fastapi.testclient import TestClient

from motionsense_app.main import create_app
from motionsense_app.settings import Settings


def test_unbuilt_root_explains_setup_and_keeps_api_available(tmp_path):
    with TestClient(create_app(Settings(root=tmp_path))) as client:
        root = client.get("/")
        assert root.status_code == 503
        assert "Cai_dat.bat" in root.text
        assert "text/html" in root.headers["content-type"]
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/unknown").json()["error"]["code"] == "NOT_FOUND"


def test_dist_serves_only_root_and_assets_not_project_or_api_fallback(tmp_path):
    dist = tmp_path / "frontend" / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("<html>MotionSense</html>", encoding="utf-8")
    (assets / "app.js").write_text("console.log('MotionSense')", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("private", encoding="utf-8")
    with TestClient(create_app(Settings(root=tmp_path))) as client:
        assert client.get("/").text == "<html>MotionSense</html>"
        assert client.get("/assets/app.js").text == "console.log('MotionSense')"
        for path in ("/api/unknown", "/secret.txt", "/frontend/dist/index.html",
                     "/assets/missing.js", "/assets/../secret.txt"):
            response = client.get(path)
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "NOT_FOUND"
