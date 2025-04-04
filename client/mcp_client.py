from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession, stdio_client, StdioServerParameters


class MCPClient:
    def __init__(self, command):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.command = command

    async def __aenter__(self):
        await self.connect_to_server()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.exit_stack.__aexit__(exc_type, exc, tb)

    async def connect_to_server(self):
        server_script_path = self.command

        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        return await self.session.initialize()

    async def get_available_tools(self):
        response = await self.session.list_tools()

        tools = response.tools

        tool_specs = []
        for tool in tools:
            tool_specs.append({
                'toolSpec': {
                    'name': tool.name,
                    'description': tool.description,
                    "inputSchema": {
                        "json": tool.inputSchema
                    }
                }
            })

        return tool_specs

    async def execute_tool(self, tool_name: str, arguments: dict):
        result = await self.session.call_tool(tool_name, arguments=arguments)
        return result

    async def cleanup(self):
        await self.exit_stack.aclose()