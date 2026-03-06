from typing import Dict, Any
from dataclasses import dataclass
from config.config import Config

config = Config()


@dataclass
class AnthropicConfig:
    # API Configuration
    anthropic_api_key: str = config.get("ANTHROPIC_API_KEY")
    base_url: str = "https://api.anthropic.com/v1"
    anthropic_version: str = "2023-06-01"
    beta_headers: str = (
        "files-api-2025-04-14,skills-2025-10-02,code-execution-2025-08-25"
    )

    # Retry Configuration
    max_retries: int = 6
    initial_backoff: float = 3.0

    # Model Configuration
    default_model: str = config.get("CLAUDE_MODEL")

    # Timeout Configuration
    timeout: float = (
        300.0  # 5 minutes (was 120) - Increased for PDF processing with skills
    )
    upload_timeout: float = 120.0  # 2 minutes (was 60) - Increased for large files

    # System Prompts
    default_system_prompt: str = (
        "You are a proactive file processing assistant with access to specialized skills and a persistent workspace.\n\n"
        "## Your Capabilities\n"
        "- You have access to file processing skills (PDF manipulation, image conversion, document editing, etc.)\n"
        "- You work in a persistent container where files you create or modify are saved across the conversation\n"
        "- When users upload files, they become available in your workspace automatically\n\n"
        "## How to Work with Files\n"
        "1. When a user uploads a file, it's provided as a document content block - you can read and analyze it immediately\n"
        "2. IMPORTANT: Uploaded files are NOT on the filesystem - they're read-only content blocks\n"
        "3. To modify an uploaded file: read its content from the document block, then create a NEW file with the changes\n"
        "4. To process or transform files, invoke the appropriate skill tool\n"
        "5. CRITICAL: To export a file, copy it to /files/output/ - this automatically assigns a file_id for download\n"
        "6. Files in /tmp/ or other directories are NOT exported and have no file_id\n"
        "7. Always complete the export step by copying to /files/output/ and mentioning the file_id\n\n"
        "## Important Guidelines\n"
        "- BE PROACTIVE: When asked to create, modify, or convert a file, invoke the skill immediately - don't just describe what you would do\n"
        "- BE DIRECT: Skip conversational confirmations like 'I can help with that' and proceed directly to action\n"
        "- BE SPECIFIC: Always mention the file_id of files you create so users know what to download\n"
        "- BE PERSISTENT: Remember that all files in the workspace persist across messages in this conversation"
    )

    # Output Configuration
    download_dir: str = "downloads"

    def get_headers(self) -> Dict[str, Any]:
        """
        Get standard HTTP headers for API requests.

        Returns:
            Dictionary of HTTP headers
        """
        return {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": self.anthropic_version,
            "anthropic-beta": self.beta_headers,
        }

    def get_headers_with_content_type(self) -> Dict[str, Any]:
        """
        Get HTTP headers including content-type for JSON requests.

        Returns:
            Dictionary of HTTP headers
        """
        headers = self.get_headers()
        headers["content-type"] = "application/json"
        return headers
