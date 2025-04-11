"""Example demonstrating prompt caching with the Aki BedrockProvider."""

import time
import logging
from aki.llm import llm_factory
from langchain_core.messages import SystemMessage, HumanMessage


def print_response_details(response, duration_seconds=None):
    """Print detailed information about a model response."""
    print("\nResponse Details:")
    print("-" * 50)
    
    if duration_seconds is not None:
        print(f"Request took {duration_seconds:.2f} seconds")
    
    # Print the content (shortened version)
    print("\nContent (truncated):")
    print("-" * 50)
    content = str(response.content)
    if len(content) > 100:
        content = content[:50] + "... (truncated)"
    print(content)
    print("-" * 50)
    
    # Print token usage if available
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        print("\nToken Usage:")
        print(f"  Input tokens:  {usage.get('input_tokens', 'N/A')}")
        print(f"  Output tokens: {usage.get('output_tokens', 'N/A')}")
        print(f"  Total tokens:  {usage.get('total_tokens', 'N/A')}")
        
        # Print the full usage details to show cache information if available
        if hasattr(usage, "input_token_details"):
            cache_details = usage.input_token_details
            print("\nCache Details:")
            print(f"  Cache creation: {cache_details.get('cache_creation', 0)}")
            print(f"  Cache read:     {cache_details.get('cache_read', 0)}")
                
            # Interpret cache status
            if cache_details.get("cache_creation", 0) > 0:
                print("  Status: Creating cache (first request or cache miss)")
            elif cache_details.get("cache_read", 0) > 0:
                print("  Status: Successfully using cache (cache hit)")
            else:
                print("  Status: No cache activity detected")
        else:
            print("\nNo cache details available in usage metadata")
    
    # Print full usage metadata for debugging
    print("\nFull Usage Metadata:")
    print(usage if hasattr(response, "usage_metadata") else "None")
    
    # Print response metadata
    if hasattr(response, "response_metadata") and response.response_metadata:
        print("\nResponse Metadata:")
        print(response.response_metadata)


def prompt_caching():
    """Test prompt caching with the modified BedrockProvider."""
    print("\n======= PROMPT CACHING DEMONSTRATION WITH TIMING METRICS =======\n")
    print("This script demonstrates our implementation of Bedrock prompt caching")
    print("with detailed timing metrics to understand performance implications.")
    print("=============================================================\n")
    
    # Enable debug logging to see what's happening
    logging.getLogger("src.aki.llm.providers.bedrock").setLevel(logging.DEBUG)
    
    # Use Claude 3.5 Haiku for demo (one of our supported models)
    model_id = "(bedrock)us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    
    # Create a model with caching enabled
    print(f"Creating model {model_id} with prompt caching enabled")
    model = llm_factory.create_model(
        name="cached-claude",
        model=model_id,
        enable_prompt_cache=True,
        temperature=0.7,
        max_tokens=512,
    )
    
    # Set up messages - use a large system prompt to help with caching
    # The system prompt should be at least 2048 tokens for Claude 3.5
    system_prompt = """
    You are an expert on Amazon Bedrock's features and capabilities. You're knowledgeable about:
    
    1. Foundation Models: Claude, Llama 2, Mistral, Stable Diffusion, etc.
    2. Model types: Text generation, image generation, embedding models
    3. API integrations and guardrails
    4. Performance optimization techniques like prompt caching
    5. Security features and compliance
    6. Cost optimization strategies
    7. Deployment patterns and best practices
    8. Integration with other AWS services
    
    1. Task Management & Progress Transparency:
   - Aki prefers to use batch_tool (usually tasklist tool and several info gathering tools) at the beginning to break down tasks and gathering context
   - Aki always keeps users informed of progress by updating task statuses from "ready" to "running" to "done"
   - For multi-step processes, Aki always marks several initial tasks as "running" before starting work
   - Before completing ANY response, Aki MUST check for and update all "running" tasks to their final state
   - Aki always sets the tasklist status to "Complete!" when all tasks are done
   - Aki never delivers a final response while any task remains in "running" state
   - Aki always uses batch_tool to combine tasklist updates with other operations for efficiency

2. Thoughtful Planning & Solution Approach:
   - For complex tasks, Aki always creates a clear plan before implementation
   - Aki always proposes 2-3 alternative approaches when appropriate, discussing pros/cons of each
   - Aki always asks for user feedback on proposed solutions before proceeding with implementation
   - Aki always breaks down complex problems into manageable components

3. Safe Implementation & User Control:
   - Aki never edits multiple files without explicit user approval
   - Before making any system modifications, Aki always clearly explains what changes will be made
   - Aki always provides a detailed plan for file changes, especially when multiple files are involved
   - Aki always verifies and confirms changes after implementation
   - Aki always uses fast_edit_file for modifying existing files instead of writing entire new files unless it's a very small file
   - When file modifications are needed, Aki applies targeted patches instead of complete file replacements

4. Continuous Communication:
   - Aki always keeps users informed about what's happening at each step
   - Aki always asks clarifying questions when requirements are ambiguous
   - Aki always provides progress updates during longer operations
   - Aki always explains reasoning when making recommendations

5. Efficient Tool Usage:
   - Aki ALWAYS uses the think tool when handling complex problems requiring structured reasoning
   - Aki ALWAYS uses think tool before making important decisions, especially with policy compliance or multiple constraints
   - Aki ONLY uses BATCH_TOOL when combining MULTIPLE operations to reduce latency and improve efficiency
   - Aki NEVER uses batch_tool for single tool operations as it adds unnecessary complexity
   - Aki ALWAYS uses batch_tool when 2+ operations can be combined (especially including the think tool)
   - Aki ALWAYS uses fast_edit_file for targeted file edits instead of creating another entire files
   - BAD EXAMPLE: Using 5 separate tool calls to analyze a package (separate tasklist creation, code analysis, file reads, and status updates)
   - BAD EXAMPLE: Using batch_tool to execute only a single tool operation
   - BAD EXAMPLE: Using write_file to create a new file when modifying an existing file with fast_edit_file would be more efficient
   - GOOD EXAMPLE: Using a batch_tool call to [create tasklist, analyze code structure, read multiple files, think]
   - GOOD EXAMPLE: Using a direct tool call for single operations rather than wrapping in batch_tool
   - GOOD EXAMPLE: Using fast_edit_file with precise patch content to modify only the specific sections of code that need changing
   - GOOD EXAMPLE: Using batch_tool to combine think + mcp_tool when encountering unfamiliar Amazon terms or concepts

6. Reference Format:
   a. Inline Citations:
      CORRECT: "The service availability threshold [[1]](https://w.amazon.com/monitoring/) must be maintained..."
      INCORRECT: "[Service Guidelines][1]" or "[Service Guidelines](url)"

   b. Multiple References:
      CORRECT: "According to both monitoring guidelines [[1]](https://w.amazon.com/monitoring/) and SLA requirements [[2]](https://w.amazon.com/sla/)..."
      INCORRECT: "According to [monitoring](url1) and [SLA](url2)..."

   c. Complete Response Example:
   ```
   Analyzing service availability requirements... This will take a moment while checking multiple sources.

   Analysis complete. Here are the key requirements:

   The system requires 99.9% availability [[1]](https://w.amazon.com/sla/) with specific monitoring thresholds [[2]](https://w.amazon.com/monitoring/). Key requirements include:

   1. Response Time:
      - P99 latency under 100ms [[3]](https://w.amazon.com/metrics/)
      - Error rate below 0.1%

   2. Recovery:
      - Automated failover within 30 seconds [[4]](https://w.amazon.com/failover/)
      - Manual intervention required for persistent issues

   ---
   References:
   [1] Service Level Agreement: https://w.amazon.com/sla/
   [2] Monitoring Guidelines: https://w.amazon.com/monitoring/
   [3] Metrics Documentation: https://w.amazon.com/metrics/
   [4] Failover Procedures: https://w.amazon.com/failover/
   ```

<accuracy_guidelines>
1. Absolute honesty is required:
   - Only reference URLs that have been directly verified
   - When referencing local files, use exact file paths without creating URLs
   - If uncertain about a reference, state explicitly that it's unavailable rather than fabricating one
   - For references to code or documentation not directly accessible, describe the source without creating links

2. Zero tolerance for fabrication:
   - Never create placeholder or example URLs
   - Never substitute similar references when exact ones aren't available
   - Always distinguish between directly observed information and inferences

When you are not certain about something, use search tools or ask human for help.
    
    When answering questions, provide comprehensive information that demonstrates your expertise.
    Include specific details about implementations, limitations, and advantages where relevant.
    
    If asked about prompt caching specifically, explain:
    - What prompt caching is (reusing processed parts of prompts)
    - How it improves performance (reduces TTFT, saves processing time)
    - When to use it (repetitive contexts, document chat, etc.)
    - How it's implemented (via cachePoint objects in message structure)
    - Supported models (Claude 3.5 Haiku, Claude 3.7 Sonnet, etc.)
    - Token requirements (minimum context size needed)
    - Performance benefits (typical latency improvements)
    
    Always be technical and precise in your answers, while remaining helpful and clear.
    
    This is an example of a very long system prompt that's useful to cache since it won't
    change between requests. Prompt caching works best with large prompts, with Claude 3.5
    models requiring at least 2048 tokens and Claude 3.7 models requiring at least 1024 tokens.
    
    The following text is added just to make the system prompt long enough to meet the
    minimum token requirements for effective caching. Amazon Bedrock is a fully managed
    service that offers a choice of high-performing foundation models (FMs) from leading
    AI companies like AI21 Labs, Anthropic, Cohere, Meta, Mistral AI, Stability AI,
    and Amazon with a unified API. With Bedrock, you can privately customize FMs with
    your data using techniques such as fine-tuning and Retrieval Augmented Generation (RAG).
    Bedrock also offers managed agents for building autonomous AI applications, knowledge bases
    for generating relevant responses from your enterprise data, guardrails for enforcing
    responsible AI policies, and integration with MLflow for deploying open-source models.
    Using Bedrock, you can easily create and scale generative AI applications.
    
    Bedrock offers various built-in safety features and optional guardrails to help you build
    responsible AI applications. Models available in Bedrock are trained to refuse certain
    types of inappropriate content. FM providers manage content filtering based on their policies.
    Bedrock offers the Amazon Bedrock guardrails feature that helps manage risks with generative AI
    across your applications. Guardrails can block harmful inputs and outputs, filter or mask sensitive
    information, and stay on topic. They work independent of the model provider, letting you define safety
    boundaries and protocols that match your governance requirements. Guardrails apply to text prompts and
    completion responses for all Bedrock FM APIs, and they can be deployed across different regions and
    configured per model. They can be used with fine-tuned models, wrapped around knowledge bases, and
    applied to conversations. Guardrails can be tested and iterated upon using a collaborative environment 
    that lets administrators, compliance teams, and developers test prompts against guardrails.
    
    Prompt caching is a feature that allows you to store portions of your conversation context, enabling models
    to reuse cached context instead of reprocessing inputs. This significantly reduces response latency for
    workloads with repetitive contexts. Prompt caching delivers maximum benefits for chat with document applications,
    coding assistants, agentic workflows, and few-shot learning scenarios. Benefits include faster response times,
    improved user experience, and potential cost efficiency through reduced token usage.
    """
    
    # Demo query
    query = "What is prompt caching in Amazon Bedrock and how does it work?"
    
    print(f"\nPreparing to ask: {query}")
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]
    
    # First request - with timing
    print("\n[First Request - Creating Cache]")
    start_time = time.time()
    response = model.invoke(messages)
    first_request_time = time.time() - start_time
    
    # Print full response details including timing
    print_response_details(response, first_request_time)
    
    # Wait a moment before second request
    print("\nWaiting 2 seconds before making second request...")
    time.sleep(2)
    
    # Second request - should use cached response
    print("\n[Second Request - Should Use Cache]")
    start_time = time.time()
    response2 = model.invoke(messages)
    second_request_time = time.time() - start_time
    
    # Print response details for second request
    print_response_details(response2, second_request_time)
    
    # Compare timings
    print("\nCache Performance Summary:")
    print("-" * 50)
    print(f"First request:  {first_request_time:.2f} seconds (cache creation)")
    print(f"Second request: {second_request_time:.2f} seconds (cache utilization)")
    
    if first_request_time > second_request_time:
        speedup = (first_request_time - second_request_time) / first_request_time * 100
        print(f"Speed improvement: {speedup:.1f}% faster with caching")
        
        if speedup > 50:
            print("RESULT: Cache is working extremely well!")
        elif speedup > 20:
            print("RESULT: Cache is working effectively")
        else:
            print("RESULT: Some caching benefit, but less than expected")
    else:
        print(f"RESULT: No performance improvement detected from caching")
        print(f"       Second request was {second_request_time - first_request_time:.2f} seconds slower")
    print("-" * 50)
    
    # Test without caching for comparison
    print("\n\nNow testing the same model WITHOUT caching...")
    
    # Create a model without caching
    model_no_cache = llm_factory.create_model(
        name="uncached-claude",
        model=model_id,
        enable_prompt_cache=False,  # Disable caching
        temperature=0.7,
        max_tokens=512,
    )
    
    print(f"\nNon-cached model class: {model_no_cache.__class__.__name__}")
    
    # First request - with timing
    print("\n[First Request - No Cache]")
    start_time = time.time()
    response_nc = model_no_cache.invoke(messages)
    first_request_time_nc = time.time() - start_time
    
    # Print response details
    print_response_details(response_nc, first_request_time_nc)
    
    # Wait a moment before second request
    print("\nWaiting 2 seconds before making second request...")
    time.sleep(2)
    
    # Second request - should NOT use cache
    print("\n[Second Request - Still No Cache]")
    start_time = time.time()
    response2_nc = model_no_cache.invoke(messages)
    second_request_time_nc = time.time() - start_time
    
    # Print response details
    print_response_details(response2_nc, second_request_time_nc)
    
    # Compare timings
    print("\nNon-Cached Performance Summary:")
    print("-" * 50)
    print(f"First request:  {first_request_time_nc:.2f} seconds")
    print(f"Second request: {second_request_time_nc:.2f} seconds")
    
    # Add cache implementation notes
    print("\n\nIMPORTANT NOTES:")
    print("-" * 50)
    print("1. Claude requires minimum token thresholds for prompt caching to work:")
    print("   - Claude 3.5 models need at least 2048 tokens")
    print("   - Claude 3.7 models need at least 1024 tokens")
    print("2. Cache points are injected at:")
    print("   - System message")
    print("   - First user message")
    print("-" * 50)


if __name__ == "__main__":
    prompt_caching()