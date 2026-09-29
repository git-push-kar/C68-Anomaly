from playwright.sync_api import sync_playwright
import time, pathlib
out = pathlib.Path("outputs/screenshots")
out.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    # Streamlit
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    print("Opening Streamlit 8502...")
    page.goto("http://localhost:8502", wait_until="networkidle", timeout=30000)
    time.sleep(5)
    page.screenshot(path=str(out / "streamlit_live.png"), full_page=True)
    print("Saved streamlit_live.png")
    # Try to click Events tab - use role tab
    try:
        page.get_by_role("tab", name="Anomaly events").click()
        time.sleep(3)
        page.screenshot(path=str(out / "streamlit_events.png"), full_page=True)
        print("Saved streamlit_events.png")
        page.get_by_role("tab", name="Follow-up chat").click()
        time.sleep(3)
        page.screenshot(path=str(out / "streamlit_chat.png"), full_page=True)
        print("Saved streamlit_chat.png")
        page.get_by_role("tab", name="Live stream").click()
        time.sleep(2)
        page.screenshot(path=str(out / "streamlit_live2.png"), full_page=True)
        print("Saved streamlit_live2.png")
    except Exception as e:
        print("Tab click failed:", e)
        page.screenshot(path=str(out / "streamlit_fallback.png"), full_page=True)

    # FastAPI docs
    page2 = browser.new_page(viewport={"width": 1280, "height": 800})
    print("Opening FastAPI 8001...")
    page2.goto("http://127.0.0.1:8001/docs", wait_until="networkidle", timeout=15000)
    time.sleep(3)
    page2.screenshot(path=str(out / "fastapi_docs.png"), full_page=True)
    print("Saved fastapi_docs.png")
    browser.close()
print("Done", list(out.glob("*.png")))
