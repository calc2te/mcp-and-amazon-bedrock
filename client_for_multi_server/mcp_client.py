from contextlib import AsyncExitStack

from mcp import StdioServerParameters, ClientSession, stdio_client


class MultiMCPClient:
    def __init__(self, servers_config: dict):
        self.server_configs = servers_config
        self.clients = {}
        self.tool_mapping = {}
        self.exit_stack = AsyncExitStack()

    async def __aenter__(self):
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close_all()

    async def connect_all(self):
        for server_name, config in self.server_configs.items():
            params = StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env=config.get("env", None)
            )
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
            read_stream, write_stream = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            self.clients[server_name] = session

    async def list_all_tools(self):
        aggregated_tools = []
        for server_name, session in self.clients.items():
            tools_response = await session.list_tools()
            for tool in tools_response.tools:
                self.tool_mapping[tool.name] = server_name
                aggregated_tools.append({
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": {
                            "json": tool.inputSchema
                        }
                    }
                })
        return aggregated_tools

    async def call_tool(self, tool_name: str, arguments: dict):
        server_id = self.tool_mapping.get(tool_name)
        session = self.clients.get(server_id)
        result = await session.call_tool(tool_name, arguments=arguments)
        return result

    async def close_all(self):
        await self.exit_stack.aclose()