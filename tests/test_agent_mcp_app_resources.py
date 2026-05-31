from src.agent_mcp.app_resources import UI_RESOURCE_URI, read_ui_resource, ui_meta
from src.agent_mcp.resources import list_resources, read_resource
from src.agent_mcp.tools import list_tools


def test_ui_resource_is_registered_with_metadata():
    resources = list_resources()
    descriptor = next(resource for resource in resources if resource["uri"] == UI_RESOURCE_URI)

    assert descriptor["mimeType"] == "text/html"
    assert descriptor["_meta"]["ui"]["resourceUri"] == UI_RESOURCE_URI
    assert descriptor["_meta"]["ui"]["permissions"] == []
    assert "script-src" in descriptor["_meta"]["ui"]["csp"]


def test_ui_resource_bundles_html_css_and_js():
    content = read_ui_resource(UI_RESOURCE_URI)

    assert content["mimeType"] == "text/html"
    assert "PyHFO Results" in content["text"]
    assert "summary-grid" in content["text"]
    assert "callServerTool" in content["text"]
    assert "/*__PYHFO_CSS__*/" not in content["text"]
    assert "/*__PYHFO_JS__*/" not in content["text"]
    assert content["_meta"] == ui_meta()


def test_resource_read_serves_ui_resource():
    response = read_resource(UI_RESOURCE_URI)

    assert response["uri"] == UI_RESOURCE_URI
    assert response["mimeType"] == "text/html"


def test_detection_and_batch_tools_advertise_ui_resource():
    tools = {tool["name"]: tool for tool in list_tools()}

    assert tools["pyhfo_detect_events"]["_meta"]["ui"]["resourceUri"] == UI_RESOURCE_URI
    assert tools["pyhfo_run_batch"]["_meta"]["ui"]["resourceUri"] == UI_RESOURCE_URI


def test_ui_metadata_blocks_external_network_by_default():
    csp = ui_meta()["ui"]["csp"]

    assert csp["default-src"] == ["'none'"]
    assert csp["connect-src"] == ["'none'"]
