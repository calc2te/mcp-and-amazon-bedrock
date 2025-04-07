import asyncio
import pprint

import boto3

from client_for_multi_server.mcp_client import MultiMCPClient

AWS_ACCESS_KEY = ''
AWS_SECRET_KEY = ''
AWS_REGION = ''

async def main():
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )

    system_prompt = "Please keep your answers very short and to the point."
    # prompt = "Please tell me top pop songs."
    prompt = "Change the file name of /Users/seobs/Documents/test/image.png to images.png"
    pprint.pprint(prompt)

    message_list = []
    message_list.append({
        "role": "user",
        "content": [
            {"text": prompt}
        ],
    })

    MCP_SERVERS_CONFIG = {
        "top-song": {
            "command": "npx",
            "args": ["-y", "@calc2te/top-song"],
        },
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/Users/seobs/Documents/test"
            ]
        }
    }

    mcp_client = await MultiMCPClient(MCP_SERVERS_CONFIG).__aenter__()

    try:
        tools = await mcp_client.list_all_tools()

        response = bedrock_client.converse(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=message_list,
            system=[{"text": system_prompt}],
            toolConfig={
                "tools": tools
            },
        )

        message_list.append(response['output']['message'])

        if response['stopReason'] == 'tool_use':

            tool_requests = response['output']['message']['content']
            for tool_request in tool_requests:

                if 'toolUse' in tool_request:
                    tool = tool_request['toolUse']

                    tool_id = tool['toolUseId']
                    tool_name = tool['name']
                    tool_input = tool['input']

                    tool_result = await mcp_client.call_tool(tool_name, tool_input)

                    message_list.append({
                        "role": "user",
                        "content": [{
                            "toolResult": {
                                "toolUseId": tool_id,
                                "content": [{"text": tool_result.content[0].text}]
                            }
                        }],
                    })

                    response = bedrock_client.converse(
                        modelId="anthropic.claude-3-haiku-20240307-v1:0",
                        messages=message_list,
                        system=[{"text": system_prompt}],
                        toolConfig={
                            "tools": tools
                        },
                    )

                    pprint.pprint(response['output']['message'])
    finally:
        await mcp_client.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())