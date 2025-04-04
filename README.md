### Before the run

1. In `client/main.py`, update the `boto3.client` section with your actual AWS credentials and region
2. In `client/main.py`, change the `mcp_file_path` to your path

### Run

```bash
PYTHONPATH=. uv run client/main.py