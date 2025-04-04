import asyncio
import pprint

import boto3

from client.mcp_client import MCPClient

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
    prompt = "Please tell me top pop songs."
    mcp_file_path = "/Users/seobs/mcp-example/server/song.py"

    message_list = []
    message_list.append({
        "role": "user",
        "content": [
            {"text": prompt}
        ],
    })

    async with MCPClient(mcp_file_path) as mcp_client:
        await mcp_client.connect_to_server()
        tools = await mcp_client.get_available_tools()

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

                async with MCPClient(mcp_file_path) as mcp_client:
                    await mcp_client.connect_to_server()
                    tool_result = await mcp_client.execute_tool(tool_name, tool_input)

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

                pprint.pprint(response)


if __name__ == "__main__":
    asyncio.run(main())