"""Example demonstrating prompt caching with tools in the Aki BedrockProvider."""

import time
import json
from typing import List, Dict, Any
from src.aki.llm import llm_factory
from pydantic import BaseModel, Field


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
    if len(content) > 300:
        content = content[:250] + "... (truncated)"
    print(content)
    print("-" * 50)
    
    # Print tool calls if any
    if hasattr(response, "tool_calls") and response.tool_calls:
        print("\nTool Calls:")
        print("-" * 50)
        for i, tool_call in enumerate(response.tool_calls):
            print(f"Tool {i+1}:")
            print(f"  Name: {tool_call.get('name', 'Unknown')}")
            print(f"  Args: {json.dumps(tool_call.get('args', {}), indent=2)}")
        print("-" * 50)
    
    # Print token usage if available
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        print(f"\nToken Usage: {usage}")
    
    # Print response metadata
    if hasattr(response, "response_metadata") and response.response_metadata:
        print("\nResponse Metadata:")
        metadata = response.response_metadata
        # Print just interesting parts to keep the output manageable
        if "stopReason" in metadata:
            print(f"  Stop Reason: {metadata['stopReason']}")
        if "metrics" in metadata and "latencyMs" in metadata["metrics"]:
            print(f"  API Latency: {metadata['metrics']['latencyMs']}ms")
        if "model_name" in metadata:
            print(f"  Model: {metadata['model_name']}")


def tool_bind_caching():
    """Test prompt caching with tools using the modified BedrockProvider."""
    print("\n======= TOOL CACHING DEMONSTRATION =======\n")
    print("This script demonstrates caching with tools in Bedrock.")
    print("We'll create several complex tools and show how caching")
    print("can improve performance for tool-enabled requests.")
    print("===========================================\n")
    
    model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    # Alternatively: model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    
    print(f"Creating models for {model_id}\n")
    
    # Define complex tools that would benefit from caching - with very verbose descriptions to ensure token threshold is met
    # Tool 1: Weather API with many parameters and detailed descriptions
    class GetWeather(BaseModel):
        """Get the current weather forecast and conditions for any location worldwide with customizable parameters.
        
        This comprehensive weather information tool provides real-time meteorological data, forecasts, and historical trends
        for any specified location globally. The tool accesses multiple weather data sources including government weather services, 
        satellite imagery, radar systems, and ground-based weather stations to compile accurate and up-to-date information.
        
        The data returned includes current conditions (temperature, humidity, wind speed and direction, precipitation, 
        atmospheric pressure, visibility, UV index), short-term forecasts (hourly predictions for the next 24 hours),
        and extended forecasts (daily predictions for up to 10 days ahead) depending on the parameters specified.
        
        The tool supports various meteorological measurement units and can provide information in multiple languages.
        Results can be customized to focus on specific weather aspects relevant to different activities and industries,
        such as aviation, marine conditions, agriculture, outdoor sports, or general public interest.
        
        Response formats are configurable to suit different applications, from simple text summaries to detailed
        data sets with all available measurements and calculations. Visual representations like weather maps, 
        radar imagery, and forecast charts are also available when the appropriate format is requested.
        
        For academic or professional uses, the tool can also provide access to historical weather patterns,
        climate data analysis, and trend information when explicitly requested through the detailed parameters.
        
        IMPORTANT NOTE: Weather predictions beyond 3 days should be considered estimates and may be subject to
        change as new meteorological data becomes available. Severe weather warnings should be verified through
        official government sources when making safety-critical decisions.
        """
        location: str = Field(
            description="The specific geographic location for which weather information is requested. This can be provided as a city and state/province/country combination (e.g., 'San Francisco, CA', 'London, UK', 'Tokyo, Japan'), geographic coordinates (latitude and longitude), ZIP/postal code, or airport code (IATA/ICAO). For large cities, adding a district or area name can improve accuracy. The more specific the location provided, the more accurate the weather data will be. If multiple locations match your query, the most populous location will be selected as the default."
        )
        units: str = Field(
            default="metric",
            description="The system of measurement units to use in the weather report. Options include: 'metric' (Celsius for temperature, kilometers for distance, mm for precipitation, hectopascals for pressure); 'imperial' (Fahrenheit for temperature, miles for distance, inches for precipitation, inches of mercury for pressure); 'scientific' (Kelvin for temperature, meters for distance, mm for precipitation, pascals for pressure); 'hybrid' (uses common combinations like Celsius with mph). Defaults to metric if not specified. Unit preferences are applied consistently across all measurements in the response."
        )
        include_forecast: bool = Field(
            default=True,
            description="Controls whether to include predictive weather forecast data in addition to current conditions. When set to true (default), the response includes hourly forecasts for the next 24 hours and daily forecasts for up to 10 days, depending on data availability for the requested location. Forecast data includes predicted temperature ranges, precipitation probability and intensity, cloud cover percentages, wind conditions, and comfort indices. When set to false, only current real-time weather conditions are returned. Forecast accuracy generally decreases for locations with limited weather infrastructure and for predictions further in the future."
        )
        language: str = Field(
            default="en",
            description="The language code specifying which language the weather information should be presented in. This applies to all textual descriptions of weather conditions, forecast summaries, and weather alerts. Supported language codes include but are not limited to: 'en' (English), 'es' (Spanish), 'fr' (French), 'de' (German), 'it' (Italian), 'pt' (Portuguese), 'ru' (Russian), 'ja' (Japanese), 'zh' (Chinese), 'ar' (Arabic), 'hi' (Hindi), 'ko' (Korean), and many others. Regional language variants like 'en-GB' or 'es-MX' are also supported for more localized terminology. Technical measurements and numerical values remain unchanged regardless of language selection."
        )
        detailed: bool = Field(
            default=False,
            description="Determines the comprehensiveness of weather data included in the response. When set to false (default), provides a streamlined summary of essential weather information suitable for general usage: basic temperature, general conditions, precipitation likelihood, and wind speed. When set to true, provides an expanded dataset with additional meteorological measurements including: detailed humidity levels, atmospheric pressure trends, visibility distance, dew point, UV index values, solar radiation intensity, cloud coverage percentages by altitude, precipitation intensity and accumulation, wind gusts in addition to sustained speeds, 'feels like' temperature accounting for humidity and wind factors, air quality indices when available, pollen counts when in season, and astronomy data (sunrise/sunset, moonrise/moonset, moon phase). The detailed option is recommended for scientific applications, specialized industries like agriculture or aviation, and weather enthusiasts."
        )
        format: str = Field(
            default="simple",
            description="Controls the formatting and organization of weather information in the response. Available options include: 'simple' (default) - provides a concise, human-readable text summary optimized for conversational contexts; 'detailed' - includes all available measurements and data points in a structured format with clear labels and grouping by weather aspect; 'visual' - includes encoded data that can be rendered as weather icons, condition maps, and charts when displayed in compatible interfaces; 'comparative' - adds historical averages and deviations from normal conditions for the location and time of year; 'technical' - emphasizes precise numerical values with meteorological terminology suitable for scientific or professional use; 'emergency' - prioritizes severe weather alerts, warnings, and safety information when applicable. The format choice affects presentation only and does not impact which data points are collected, which is controlled by the 'detailed' parameter."
        )
        alerts: bool = Field(
            default=True,
            description="Determines whether to include weather warnings, advisories, and emergency alerts issued by meteorological authorities for the specified location. When enabled, provides details about active severe weather events (hurricanes, floods, storms, extreme temperatures, etc.), including severity level, affected areas, expected duration, and safety recommendations. Critical alerts are always included regardless of this setting when they represent immediate life-safety concerns."
        )
        historical_comparison: bool = Field(
            default=False,
            description="When enabled, includes comparative data showing how current conditions and forecasts compare to historical averages for the location during the same calendar period. This includes temperature anomalies, precipitation deviations from normal, and identification of unusual or record-breaking conditions. Useful for understanding weather patterns in context of typical climate expectations."
        )
        altitude_specific: bool = Field(
            default=False,
            description="When enabled, tailors weather data to account for specific elevation effects, particularly relevant in mountainous regions where conditions can vary dramatically with altitude changes. Includes additional data points on atmospheric conditions at different elevations when available."
        )
        agriculture_focus: bool = Field(
            default=False,
            description="Enhances response with agriculture-relevant weather factors including soil temperature and moisture data, growing degree days, frost/freeze probabilities, crop-specific weather indices, and extended seasonal outlooks when available. Primarily useful for farming and agricultural planning applications."
        )
        marine_data: bool = Field(
            default=False,
            description="When enabled for coastal locations or marine environments, includes additional oceanic and coastal weather data: wave heights and periods, tide information, water temperatures, marine navigation conditions, and coastal-specific weather phenomena. Particularly useful for boating, fishing, and beach activities."
        )

    # Tool 2: Translation tool with many languages and options
    class Translate(BaseModel):
        """Translate text between languages with advanced options."""
        text: str = Field(description="The text to translate")
        source_language: str = Field(description="Source language code (e.g., 'en', 'es', 'fr')")
        target_language: str = Field(description="Target language code (e.g., 'en', 'es', 'fr')")
        formality: str = Field(
            default="default",
            description="Formality level: 'formal', 'informal', or 'default'"
        )
        preserve_formatting: bool = Field(
            default=True,
            description="Whether to preserve formatting in the translation"
        )
        regional_dialect: str = Field(
            default="standard",
            description="Regional dialect variant for the target language"
        )
        glossary_terms: Dict[str, str] = Field(
            default_factory=dict,
            description="Custom glossary terms mapping for domain-specific translations"
        )
        domain: str = Field(
            default="general",
            description="Content domain: 'general', 'technical', 'medical', 'legal', etc."
        )

    # Tool 3: Database query tool with complex schema
    class QueryDatabase(BaseModel):
        """Query a relational database with complex filtering."""
        table: str = Field(description="Table name to query")
        fields: List[str] = Field(description="List of fields to retrieve")
        filters: Dict[str, Any] = Field(
            default_factory=dict,
            description="Filter conditions as field-value pairs"
        )
        sort_by: str = Field(
            default="id",
            description="Field to sort results by"
        )
        sort_direction: str = Field(
            default="asc",
            description="Sort direction: 'asc' or 'desc'"
        )
        limit: int = Field(
            default=100,
            description="Maximum number of results to return"
        )
        offset: int = Field(
            default=0,
            description="Number of results to skip"
        )
        join_tables: List[Dict[str, str]] = Field(
            default_factory=list,
            description="Tables to join with their join conditions"
        )
        group_by: List[str] = Field(
            default_factory=list,
            description="Fields to group results by"
        )
        having: Dict[str, Any] = Field(
            default_factory=dict,
            description="Having conditions for grouped results"
        )
        distinct: bool = Field(
            default=False,
            description="Whether to return distinct results"
        )
    
    # Create a cached model with the tools
    print("Creating model with prompt caching and tools enabled (max 3 cache points)")
    tools = [GetWeather, Translate, QueryDatabase]
    cached_model = llm_factory.create_model(
        name="cached-claude-tools",
        model=f"(bedrock){model_id}",
        enable_prompt_cache=True,
        max_cache_points=3,  # Use our new parameter
        temperature=0.2,
        max_tokens=1000,
    )
    
    # Bind tools to the model
    cached_model_with_tools = cached_model.bind_tools(tools)
    print(f"\nCached model class: {cached_model_with_tools.__class__.__name__}")
    
    # Create a model without caching but with the same tools
    print("\nCreating comparison model without caching")
    uncached_model = llm_factory.create_model(
        name="uncached-claude-tools",
        model=f"(bedrock){model_id}",
        enable_prompt_cache=False,  # Explicitly disable caching
        temperature=0.2,
        max_tokens=1000,
    )
    
    # Bind tools to the uncached model
    uncached_model_with_tools = uncached_model.bind_tools(tools)
    print(f"Uncached model class: {uncached_model_with_tools.__class__.__name__}")
    
    # Query to test with - asking for weather information
    query = "What's the current weather in Seattle, WA and Tokyo, Japan? Translate the weather description for Tokyo to Spanish."
    print(f"\nQuery: {query}")
    
    # Test cached model first request
    print("\n[First Request With Caching - Creating Cache]")
    start_time = time.time()
    response = cached_model_with_tools.invoke(query)
    first_request_time = time.time() - start_time
    print_response_details(response, first_request_time)
    
    # Wait a moment before second request
    print("\nWaiting 2 seconds before making second request...")
    time.sleep(2)
    
    # Test cached model second request - should use cache
    print("\n[Second Request With Caching - Should Use Cache]")
    start_time = time.time()
    response2 = cached_model_with_tools.invoke(query)
    second_request_time = time.time() - start_time
    print_response_details(response2, second_request_time)
    
    # Compare timings for cached model
    print("\nCached Model Performance Summary:")
    print("-" * 50)
    print(f"First request:  {first_request_time:.2f} seconds (cache creation)")
    print(f"Second request: {second_request_time:.2f} seconds (cache utilization)")
    
    if first_request_time > second_request_time:
        speedup = (first_request_time - second_request_time) / first_request_time * 100
        print(f"Speed improvement: {speedup:.1f}% faster with caching")
    else:
        print(f"No performance improvement detected from caching")
        print(f"Second request was {second_request_time - first_request_time:.2f} seconds slower")
    print("-" * 50)
    
    # Now test the uncached model
    print("\n\nTesting without caching for comparison:")
    
    # First request with uncached model
    print("\n[First Request Without Caching]")
    start_time = time.time()
    response_uc = uncached_model_with_tools.invoke(query)
    first_request_time_uc = time.time() - start_time
    print_response_details(response_uc, first_request_time_uc)
    
    # Wait a moment before second request
    print("\nWaiting 2 seconds before making second request...")
    time.sleep(2)
    
    # Second request with uncached model
    print("\n[Second Request Without Caching]")
    start_time = time.time()
    response2_uc = uncached_model_with_tools.invoke(query)
    second_request_time_uc = time.time() - start_time
    print_response_details(response2_uc, second_request_time_uc)
    
    # Compare timings for uncached model
    print("\nUncached Model Performance Summary:")
    print("-" * 50)
    print(f"First request:  {first_request_time_uc:.2f} seconds")
    print(f"Second request: {second_request_time_uc:.2f} seconds")
    print("-" * 50)
    
    # Print final comparison
    print("\nFinal Comparison:")
    print("-" * 50)
    print("Cached Model:")
    print(f"  First request:  {first_request_time:.2f} seconds")
    print(f"  Second request: {second_request_time:.2f} seconds")
    print("Uncached Model:")
    print(f"  First request:  {first_request_time_uc:.2f} seconds")
    print(f"  Second request: {second_request_time_uc:.2f} seconds")
    print("-" * 50)
    
    # Add implementation notes
    print("\nIMPORTANT NOTES:")
    print("-" * 50)
    print("1. Tool caching adds a cachePoint object to the tools array")
    print("2. Message caching adds up to 3 additional cache points based on:")
    print("   - System prompts > 2000 tokens get priority")
    print("   - Other messages > 1000 tokens in order of size")
    print("3. Token estimation is performed to find the largest messages")
    print("4. Cache points are injected into message content arrays")
    print("5. Total across all contexts: max 4 cache points (1 for tools + 3 for messages)")
    print("-" * 50)


if __name__ == "__main__":
    tool_bind_caching()