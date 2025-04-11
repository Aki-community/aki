"""Amazon Bedrock provider for Aki."""

import logging
import boto3
import json
from typing import Dict, List, Optional, Set, Any, Tuple, Iterator, Union, Sequence, Callable, Literal
from botocore.client import BaseClient, Config

from langchain_aws import ChatBedrockConverse
from langchain_aws.chat_models.bedrock_converse import _messages_to_bedrock, _snake_to_camel_keys, _parse_response, _parse_stream_event
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import TypeBaseModel

from ...config import get_env_file, get_config_value
from ..capabilities import ModelCapability
from .base import LLMProvider
from .. import token_counter

logger = logging.getLogger(__name__)

class CachePointInjector:
    """Helper to inject cache points into messages for Bedrock models."""
    
    @staticmethod
    def add_cache_point_to_messages(messages: List[Dict], system: List[Dict], max_cache_points: int = 3) -> tuple:
        """Add cachePoint to messages based on size and importance.
        
        Args:
            messages: List of message dictionaries from Bedrock
            system: Optional system message (string or dict)
            max_cache_points: Maximum number of cache points to add (default: 3)
            
        Returns:
            Tuple of (modified messages list, modified system)
        """
        # Prepare list of message info including system if present
        message_info = []
        
        # Add system message to consideration if it exists
        system_token_size = CachePointInjector._estimate_token_size(system[0])

        # Add system to message info with special index -1
        message_info.append({"index": -1, "size": system_token_size, "is_system": True})
        
        # Process regular messages
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            is_system = (role == "system")
            
            # Estimate token size using tiktoken
            token_size = CachePointInjector._estimate_token_size(msg)
            message_info.append({"index": i, "size": token_size, "is_system": is_system})
        
        # Sort messages by priority (system first, then by size)
        message_info.sort(key=lambda x: (-int(x["is_system"]), -x["size"]))
        
        # Determine which messages to add cache points to (up to max_cache_points)
        cache_indices = set()
        for info in message_info[:max_cache_points]:
            # Only add cache point if message is large enough
            min_size = 1500
            if info["size"] >= min_size:
                cache_indices.add(info["index"])
        
        # Process system message if it's selected for caching
        modified_system = system
        if -1 in cache_indices and system is not None:
            # Only convert to dict and add cache point if it's a string
            if "cachePoint" not in modified_system:
                modified_system.append({"cachePoint": {"type": "default"}})
        
        # Add cache points to selected regular messages
        result = []
        for i, msg in enumerate(messages):
            new_msg = dict(msg)  # Create a copy
            
            # Add cache point if this message is selected
            if i in cache_indices:
                new_msg = CachePointInjector._add_cache_point(new_msg)
                logger.debug(f"Added cache point to message with role: {new_msg.get('role', 'unknown')}")
            
            result.append(new_msg)
        
        return result, modified_system
    
    @staticmethod
    def _estimate_token_size(msg: Dict) -> int:
        """Estimate token size of a message."""
        # Extract the content - handle different formats
        if "content" not in msg:
            if "text" in msg:
                content = msg["text"]
            else:
                return 0
        else:
            content = msg["content"]
        
        # If content is a string, count tokens directly
        if isinstance(content, str):
            return token_counter.str_token_counter(content)
        
        # If content is a list of content blocks, sum up their sizes
        elif isinstance(content, list):
            total = 0
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    total += token_counter.str_token_counter(item["text"])
                elif isinstance(item, dict):
                    # For non-text blocks, estimate based on JSON representation
                    total += token_counter.str_token_counter(json.dumps(item))
                elif isinstance(item, str):
                    total += token_counter.str_token_counter(item)
            return total
        
        # Default fallback - convert to string and count
        return token_counter.str_token_counter(json.dumps(content))
    
    @staticmethod
    def _add_cache_point(msg: Dict) -> Dict:
        """Add a cache point to a single message."""
        # Make a deep copy
        new_msg = dict(msg)
        
        # Handle different content formats
        if "content" not in new_msg:
            new_msg["content"] = []
        elif isinstance(new_msg["content"], str):
            # Convert string content to list with text item
            new_msg["content"] = [{"text": new_msg["content"]}]
        elif not isinstance(new_msg["content"], list):
            # Convert other content to list
            new_msg["content"] = [new_msg["content"]]
            
        # Add cache point if not already present
        content_list = new_msg["content"]
        cache_point = {"cachePoint": {"type": "default"}}
        
        # Only add if not already present
        if not any(isinstance(item, dict) and "cachePoint" in item for item in content_list):
            content_list.append(cache_point)
        
        # Print debug info
        logger.debug(f"Added cache point to message with role: {new_msg.get('role', 'unknown')}")
            
        return new_msg


class CachingBedrockConverse(ChatBedrockConverse):
    """ChatBedrockConverse with prompt caching support."""
    
    max_cache_points: int = 3  # Properly declare field for Pydantic model
    
    def __init__(self, **kwargs):
        """Initialize with caching configuration.
        
        Args:
            max_cache_points: Maximum number of cache points to add (default: 3)
            All other kwargs are passed to the parent class
        """
        # Get max_cache_points to pass to parent class init
        max_points = kwargs.pop("max_cache_points", 3)
        super().__init__(**kwargs)
        # Set the field after parent initialization
        object.__setattr__(self, "max_cache_points", max_points)
    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], TypeBaseModel, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "any"]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Override bind_tools to add cache point to the tools config."""
        # Use the parent's method to format the tools and create the base model
        model_with_tools = super().bind_tools(tools, tool_choice=tool_choice, **kwargs)
        
        # Get the current params to modify them
        if hasattr(model_with_tools, "kwargs") and "toolConfig" in model_with_tools.kwargs:
            # Add cache point to the tools list
            if "tools" in model_with_tools.kwargs["toolConfig"]:
                # Check if there's already a cache point
                tools_list = model_with_tools.kwargs["toolConfig"]["tools"]
                has_cache_point = any(
                    isinstance(tool, dict) and "cachePoint" in tool 
                    for tool in tools_list
                )
                
                # Add cache point if needed
                if not has_cache_point:
                    logger.info("Adding cache point to tools configuration")
                    tools_list.append({"cachePoint": {"type": "default"}})
        
        return model_with_tools
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override _generate to add cache points to messages."""
        # Get bedrock messages using the original function
        bedrock_messages, system = _messages_to_bedrock(messages)
        
        # Inject cache points to both regular messages and system
        bedrock_messages, system = CachePointInjector.add_cache_point_to_messages(
            bedrock_messages,
            system=system,
            max_cache_points=self.max_cache_points
        )
        
        # Continue with regular process - similar to parent class
        logger.debug(f"input message to bedrock: {bedrock_messages}")
        logger.debug(f"System message to bedrock: {system}")
        params = self._converse_params(
            stop=stop, **_snake_to_camel_keys(kwargs, excluded_keys={"inputSchema", "properties", "thinking"})
        )
        logger.debug(f"Input params: {params}")
        logger.info("Using Bedrock Converse API to generate response with caching enabled")
        response = self.client.converse(
            messages=bedrock_messages, system=system, **params
        )
        logger.debug(f"Response from Bedrock: {response}")
        response_message = _parse_response(response)
        response_message.response_metadata["model_name"] = self.model_id
        return ChatResult(generations=[ChatGeneration(message=response_message)])
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Override _stream to add cache points to messages."""
        # Get bedrock messages using the original function
        bedrock_messages, system = _messages_to_bedrock(messages)
        
        # Inject cache points to both regular messages and system
        bedrock_messages, system = CachePointInjector.add_cache_point_to_messages(
            bedrock_messages,
            system=system,
            max_cache_points=self.max_cache_points
        )
        
        # Continue with regular process - similar to parent class
        params = self._converse_params(
            stop=stop, **_snake_to_camel_keys(kwargs, excluded_keys={"inputSchema", "properties", "thinking"})
        )
        response = self.client.converse_stream(
            messages=bedrock_messages, system=system, **params
        )
        added_model_name = False
        for event in response["stream"]:
            if message_chunk := _parse_stream_event(event):
                if (
                    hasattr(message_chunk, "usage_metadata")
                    and message_chunk.usage_metadata
                    and not added_model_name
                ):
                    message_chunk.response_metadata["model_name"] = self.model_id
                    added_model_name = True
                generation_chunk = ChatGenerationChunk(message=message_chunk)
                if run_manager:
                    run_manager.on_llm_new_token(
                        generation_chunk.text, chunk=generation_chunk
                    )
                yield generation_chunk


def validate_bedrock_access(session: boto3.Session) -> bool:
    """Validate Bedrock access by attempting to list foundation models.

    Args:
        session: boto3.Session to validate

    Returns:
        bool: True if the session has valid Bedrock access, False otherwise
    """
    try:
        # Get region from environment or use default
        region = get_config_value("AWS_DEFAULT_REGION", "us-west-2")
        client = session.client("bedrock", region_name=region)
        response = client.list_foundation_models()
        models = response["modelSummaries"]
        # TODO: need to verify models that we use
        logger.debug(
            f"Successfully validated Bedrock access. Found {len(models)} foundation models."
        )
        return True
    except Exception as e:
        logger.debug(f"Bedrock access validation failed: {str(e)}")
        return False


def create_session_from_env(env_vars: dict) -> Tuple[Optional[boto3.Session], str]:
    """Create a boto3 session from environment variables.

    Args:
        env_vars: Dictionary containing AWS credentials

    Returns:
        Tuple[Optional[boto3.Session], str]: The created session (or None) and a message
    """
    aws_access_key = env_vars.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = env_vars.get("AWS_SECRET_ACCESS_KEY")

    if not aws_access_key or not aws_secret_key:
        return None, "Missing required AWS credentials"

    session = boto3.Session(
        aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
    )

    if validate_bedrock_access(session):
        return session, "Successfully validated credentials"
    return None, "Failed to validate Bedrock access"


class BedrockProvider(LLMProvider):
    """Provider for Amazon Bedrock models."""

    def __init__(self):
        self._model_capabilities = self.capabilities
        self._logger = logging.getLogger(__name__)
        self._client = None

    def _create_client(self) -> BaseClient:
        """Create a Bedrock client with credential discovery.

        Attempts to create a client using credentials in the following order:
        1. Credentials from ~/.aki/.env file
        2. AWS profile 'aki' (if it exists)

        Returns:
            BaseClient: A configured boto3 bedrock-runtime client
        """
        try:
            # Get region from environment with fallback to default
            region = get_config_value("AWS_DEFAULT_REGION", "us-west-2")
            logger.debug(f"Using AWS region: {region}")
            
            config_obj = Config(
                read_timeout=20000,
                connect_timeout=20000,
                retries={"max_attempts": 5, "mode": "adaptive"},
            )

            # 1. Try loading credentials from ~/.aki/.env
            logger.debug("Attempting to use credentials from ~/.aki/.env")
            env_path = get_env_file()
            if env_path.exists():
                try:
                    # Load credentials from env file
                    env_vars = {
                        "AWS_ACCESS_KEY_ID": get_config_value("AWS_ACCESS_KEY_ID"),
                        "AWS_SECRET_ACCESS_KEY": get_config_value("AWS_SECRET_ACCESS_KEY")
                    }
                    
                    if env_vars["AWS_ACCESS_KEY_ID"] and env_vars["AWS_SECRET_ACCESS_KEY"]:
                        session = boto3.Session(
                            aws_access_key_id=env_vars["AWS_ACCESS_KEY_ID"],
                            aws_secret_access_key=env_vars["AWS_SECRET_ACCESS_KEY"]
                        )
                        
                        if validate_bedrock_access(session):
                            logger.debug("Successfully validated credentials from ~/.aki/.env")
                            return session.client(
                                "bedrock-runtime", region_name=region, config=config_obj
                            )
                        logger.debug("Failed to validate credentials from ~/.aki/.env")
                except Exception as e:
                    logger.warning(f"Error using credentials from ~/.aki/.env: {e}")
            else:
                logger.debug("No ~/.aki/.env file found")

            # 2. Try profile-based authentication
            logger.debug("Attempting to use aki profile authentication")
            try:
                session = boto3.Session(profile_name="aki")
                if validate_bedrock_access(session):
                    logger.debug("Successfully validated aki profile credentials")
                    return session.client(
                        "bedrock-runtime", region_name=region, config=config_obj
                    )
                logger.debug("Failed to validate aki profile credentials")
            except Exception as e:
                logger.debug(f"Error using aki profile: {e}")

            # If we get here, neither method worked
            raise ValueError(
                "Failed to create Bedrock client. Could not find valid credentials in:\n"
                "1. ~/.aki/.env file\n"
                "2. AWS profile 'aki'\n"
            )

        except Exception as e:
            self._logger.error(f"Failed to create Bedrock client: {e}")
            raise

    def _get_client(self) -> BaseClient:
        """Get or create a Bedrock client."""
        if not self._client:
            self._client = self._create_client()
        return self._client

    def _get_model_config(
        self, model: str, tools: Optional[List] = None, **kwargs
    ) -> Dict[str, Any]:
        """Get configuration for a specific model.

        This is a protected method that can be overridden by subclasses
        to implement custom model configuration logic.

        Args:
            model: The model ID
            tools: Optional list of tools for tool-calling models
            **kwargs: Additional model configuration parameters

        Returns:
            Dict[str, Any]: Model configuration dictionary
        """
        # Extract extended reasoning settings if present
        enable_reasoning = kwargs.pop("enable_reasoning", False)
        budget_tokens = kwargs.pop("budget_tokens", 1024)

        # Base model configuration
        model_kwargs = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", 8192),
            "temperature": kwargs.pop("temperature", 0.6),
        }

        # Add extended reasoning configuration if enabled for supported models
        if (
            enable_reasoning
            and ModelCapability.EXTENDED_REASONING
            in self._model_capabilities.get(model, set())
        ):
            self._logger.debug(
                f"Enabling extended reasoning for {model} with budget {budget_tokens}"
            )
            # Claude 3.7 requires temperature=1.0 for extended reasoning
            model_kwargs["temperature"] = 1.0
            # Increase max_tokens to accommodate reasoning budget plus regular response
            model_kwargs["max_tokens"] = budget_tokens + model_kwargs.get(
                "max_tokens", 8192
            )
            # Add reasoning configuration to model_kwargs
            model_kwargs["additional_model_request_fields"] = {
                "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
                "anthropic_beta": ["token-efficient-tools-2025-02-19"],
            }

        # Add any remaining kwargs
        for key, value in kwargs.items():
            model_kwargs[key] = value

        return model_kwargs

    def create_model(
        self, name: str, model: str, tools: Optional[List] = None, **kwargs
    ) -> BaseChatModel:
        """Create a new Bedrock chat model with optional caching.

        Args:
            name: A name for the model instance
            model: The Bedrock model ID
            tools: Optional list of tools for tool-calling models
            **kwargs: Additional arguments to pass to the model, including:
                enable_prompt_cache (bool): Enable prompt caching if supported
                max_cache_points (int): Maximum number of cache points to add (default: 3)

        Returns:
            A configured ChatBedrockConverse instance
        """
        # Extract caching parameters
        enable_prompt_cache = kwargs.pop("enable_prompt_cache", False)
        max_cache_points = kwargs.pop("max_cache_points", 3)
        
        client = self._get_client()
        model_kwargs = self._get_model_config(model, tools, **kwargs)
        model_kwargs["client"] = client
        
        try:
            # Check if caching is enabled and supported
            use_caching = (
                enable_prompt_cache and
                ModelCapability.PROMPT_CACHING in self._model_capabilities.get(model, set())
            )
            
            if use_caching:
                self._logger.debug(f"Creating model {model} with prompt caching enabled (max {max_cache_points} cache points)")
                llm = CachingBedrockConverse(name=name, max_cache_points=max_cache_points, **model_kwargs)
            else:
                llm = ChatBedrockConverse(name=name, **model_kwargs)
                
            # Add tools if provided and supported
            if (
                tools
                and len(tools) > 0
                and ModelCapability.TOOL_CALLING
                in self._model_capabilities.get(model, set())
            ):
                llm = llm.bind_tools(tools)

            return llm
        except Exception as e:
            self._logger.error(f"Failed to create Bedrock model {model}: {e}")
            raise

    def list_models(self) -> List[str]:
        """List all available models from this provider.

        Returns:
            List of model IDs that can be used with this provider
        """
        return list(self._model_capabilities.keys())

    @property
    def name(self) -> str:
        """Get the name of this provider."""
        return "bedrock"

    @property
    def capabilities(self) -> Dict[str, Set[ModelCapability]]:
        return {
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0": {
                ModelCapability.TEXT_TO_TEXT,
                ModelCapability.IMAGE_TO_TEXT,
                ModelCapability.TOOL_CALLING,
                ModelCapability.STRUCTURED_OUTPUT,
                ModelCapability.EXTENDED_REASONING,
                ModelCapability.PROMPT_CACHING,
            },
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {
                ModelCapability.TEXT_TO_TEXT,
                ModelCapability.IMAGE_TO_TEXT,
                ModelCapability.TOOL_CALLING,
                ModelCapability.STRUCTURED_OUTPUT,
            },
            "us.anthropic.claude-3-5-haiku-20241022-v1:0": {
                ModelCapability.TEXT_TO_TEXT,
                ModelCapability.IMAGE_TO_TEXT,
                ModelCapability.TOOL_CALLING,
                ModelCapability.STRUCTURED_OUTPUT,
                ModelCapability.PROMPT_CACHING,
            },
            "us.anthropic.claude-3-5-sonnet-20240620-v1:0": {
                ModelCapability.TEXT_TO_TEXT,
                ModelCapability.IMAGE_TO_TEXT,
                ModelCapability.TOOL_CALLING,
                ModelCapability.STRUCTURED_OUTPUT,
            },
            "stability.stable-image-ultra-v1:0": {ModelCapability.TEXT_TO_IMAGE},
            "stability.stable-image-core-v1:0": {ModelCapability.TEXT_TO_IMAGE},
            "us.deepseek.r1-v1:0": {ModelCapability.TEXT_TO_TEXT},
        }
