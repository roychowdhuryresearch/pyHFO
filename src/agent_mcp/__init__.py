__all__ = ["PyHFOMCPServer"]


def __getattr__(name):
    if name == "PyHFOMCPServer":
        from src.agent_mcp.server import PyHFOMCPServer

        return PyHFOMCPServer
    raise AttributeError(name)
