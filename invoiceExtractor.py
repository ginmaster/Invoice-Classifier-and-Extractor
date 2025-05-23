#!/usr/bin/env python3
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

# --- Load Extraction Schema ------------------------------------------------------------
def load_extraction_schema(path: str = "extraction_schema.json") -> dict[str, Any]:
    """Load field extraction schema from JSON file.
    
    The schema defines the structure and types of fields to extract from invoices,
    including nested objects and arrays for line items.
    
    Args:
        path: Path to the extraction schema JSON file. Defaults to 'extraction_schema.json'.
        
    Returns:
        Dictionary containing field definitions with 'fields' key.
        
    Raises:
        FileNotFoundError: If schema file doesn't exist at specified path.
        json.JSONDecodeError: If file contains invalid JSON syntax.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Configuration -------------------------------------------------------------------
@dataclass(kw_only=True)
class Settings:
    """Configuration settings for invoice extraction.
    
    Attributes:
        endpoint: Azure Content Understanding API endpoint URL.
        api_version: API version to use.
        subscription_key: Subscription key for authentication (optional if using AAD).
        aad_token: Azure Active Directory token (optional if using subscription key).
        analyzer_id: Custom analyzer identifier.
        classifier_id: Document classifier identifier.
        file_location: Path to invoice file or URL.
        extraction_schema_path: Path to field extraction schema file.
        http_timeout: HTTP request timeout in seconds.
        retries: Number of retry attempts for failed requests.
        backoff_factor: Exponential backoff factor for retries.
        
    Raises:
        ValueError: If neither subscription_key nor aad_token is provided.
    """
    endpoint: str
    api_version: str = "2025-05-01-preview"
    subscription_key: Optional[str] = None
    aad_token: Optional[str] = None
    analyzer_id: str = ""
    classifier_id: str = ""
    file_location: str = "invoice.pdf"
    extraction_schema_path: str = "extraction_schema.json"
    http_timeout: float = 30.0
    retries: int = 5
    backoff_factor: float = 0.5

    def __post_init__(self):
        if not self.subscription_key and not self.aad_token:
            raise ValueError("Either 'subscription_key' or 'aad_token' must be provided")

    @property
    def token_provider(self) -> Optional[Callable[[], str]]:
        """Get Azure Active Directory token provider for authentication.
        
        Returns:
            Callable that returns current AAD token, or None if using subscription key auth.
        """
        if self.aad_token:
            return AADTokenProvider(self.aad_token)
        return None

# --- Token Provider ---------------------------------------------------------------
class AADTokenProvider:
    """Azure Active Directory token provider with automatic refresh support.
    
    Manages AAD tokens and handles expiration by tracking token lifetime.
    Currently requires manual token refresh implementation.
    
    Args:
        initial_token: Initial AAD bearer token for authentication.
        expires_in: Token validity period in seconds. Defaults to 3300 (55 minutes).
    """
    def __init__(self, initial_token: str, expires_in: int = 3300):
        self._token = initial_token
        self._expires_at = time.time() + expires_in

    def __call__(self) -> str:
        """Get current valid AAD token.
        
        Checks token expiration and triggers refresh if within 60 seconds of expiry.
        Note: Token refresh logic needs to be implemented.
        
        Returns:
            Valid AAD bearer token string.
        """
        if time.time() > self._expires_at - 60:
            # TODO: Refresh token here
            pass
        return self._token

# --- Client -------------------------------------------------------------------------
class AzureContentUnderstandingClient:
    """Client for Azure Content Understanding APIs.
    
    Provides methods for managing classifiers and analyzers, and performing
    document classification and field extraction operations.
    
    Args:
        endpoint: Azure Content Understanding service endpoint URL.
        api_version: API version for requests.
        subscription_key: Subscription key for authentication.
        token_provider: Callable that returns an AAD token.
        x_ms_useragent: User agent string for requests.
        http_timeout: Request timeout in seconds.
        retries: Number of retry attempts for failed requests.
        backoff_factor: Exponential backoff factor for retries.
        
    Raises:
        ValueError: If required parameters are missing.
    """
    def __init__(
        self,
        endpoint: str,
        api_version: str,
        subscription_key: Optional[str],
        token_provider: Optional[Callable[[], str]],
        x_ms_useragent: str = "invoice-extractor",
        http_timeout: float = 30.0,
        retries: int = 5,
        backoff_factor: float = 0.5,
    ):
        if not endpoint or not api_version or (not subscription_key and token_provider is None):
            raise ValueError("Endpoint, API version and credential provider must be set")
        self._endpoint = endpoint.rstrip("/")
        self._api_version = api_version
        self._subscription_key = subscription_key
        self._token_provider = token_provider
        self._timeout = http_timeout
        self._retry_total = retries
        self._backoff = backoff_factor
        self._logger = logging.getLogger(__name__)
        self._headers = self._get_headers()
        self._session = self._create_session()

    def _get_headers(self) -> dict[str, str]:
        """Build HTTP request headers with authentication credentials.
        
        Adds either subscription key or bearer token based on configured auth method.
        
        Returns:
            Dictionary containing x-ms-useragent and authentication headers.
        """
        token = self._token_provider()() if self._token_provider else None
        headers = {"x-ms-useragent": "invoice-extractor"}
        if self._subscription_key:
            headers["Ocp-Apim-Subscription-Key"] = self._subscription_key
        elif token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _create_session(self) -> requests.Session:
        """Create HTTP session with automatic retry and backoff handling.
        
        Configures exponential backoff for transient failures and rate limiting.
        Retries on status codes: 429 (rate limit), 500, 502, 503, 504.
        
        Returns:
            Configured requests.Session instance with retry adapter.
        """
        session = requests.Session()
        retries = Retry(
            total=self._retry_total,
            backoff_factor=self._backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "POST", "DELETE", "OPTIONS", "TRACE"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _refresh_headers(self):
        """Update authentication headers with current credentials.
        
        Called before each request to ensure fresh authentication tokens.
        """
        self._headers = self._get_headers()

    # Classifier operations
    def list_classifiers(self) -> list[dict[str, Any]]:
        """List all available document classifiers in the workspace.
        
        Returns:
            List of classifier objects containing id, status, and description.
        """
        self._refresh_headers()
        url = f"{self._endpoint}/contentunderstanding/classifiers?api-version={self._api_version}"
        resp = self._session.get(url, headers=self._headers, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json().get("value", [])

    def create_or_update_classifier(self, classifier_id: str, analyzer_id: str):
        """Create or update a document classifier for invoice/receipt categorization.
        
        Configures a classifier with auto split mode and invoice category linked
        to the specified analyzer. Handles 409 conflicts gracefully for updates.
        
        Args:
            classifier_id: Unique identifier for the classifier.
            analyzer_id: ID of analyzer to use for invoice category documents.
            
        Returns:
            HTTP response object with operation result.
        """
        self._refresh_headers()
        url = f"{self._endpoint}/contentunderstanding/classifiers/{classifier_id}?api-version={self._api_version}"
        body = {"description": "Invoice/Receipt", "splitMode": "auto", "categories": {"invoice": {"analyzerId": analyzer_id}}}
        resp = self._session.put(url, headers=self._headers, json=body, timeout=self._timeout)
        
        # Handle 409 Conflict - classifier already exists, which is fine for updates
        if resp.status_code == 409:
            self._logger.info(f"Classifier '{classifier_id}' already exists, update completed")
            return resp
        
        resp.raise_for_status()
        return resp

    def delete_classifier(self, classifier_id: str):
        """Delete an existing document classifier.
        
        Args:
            classifier_id: ID of the classifier to delete.
            
        Returns:
            HTTP response object with deletion status.
        """
        self._refresh_headers()
        url = f"{self._endpoint}/contentunderstanding/classifiers/{classifier_id}?api-version={self._api_version}"
        resp = self._session.delete(url, headers=self._headers, timeout=self._timeout)
        resp.raise_for_status()
        return resp

    # Analyzer operations
    def list_analyzers(self) -> list[dict[str, Any]]:
        """List all custom field extraction analyzers in the workspace.
        
        Returns:
            List of analyzer objects containing id, status, and field schema.
        """
        self._refresh_headers()
        url = f"{self._endpoint}/contentunderstanding/analyzers?api-version={self._api_version}"
        resp = self._session.get(url, headers=self._headers, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json().get("value", [])

    def create_or_update_analyzer(self, analyzer_id: str, field_schema: dict[str, Any], description: str = "Custom Invoice Analyzer"):
        """Create or update a custom analyzer for structured field extraction.
        
        Builds on prebuilt-documentAnalyzer base model with custom field schema.
        Handles 409 conflicts gracefully for updates. Logs detailed error info on failure.
        
        Args:
            analyzer_id: Unique identifier for the analyzer.
            field_schema: Schema dict with 'fields' key containing field definitions.
            description: Human-readable description of analyzer purpose.
            
        Returns:
            HTTP response object with operation result.
        """
        self._refresh_headers()
        url = f"{self._endpoint}/contentunderstanding/analyzers/{analyzer_id}?api-version={self._api_version}"
        body = {
            "description": description,
            "baseAnalyzerId": "prebuilt-documentAnalyzer",
            "fieldSchema": {"name": "Invoice", "fields": field_schema.get("fields", {})},
        }
        resp = self._session.put(url, headers=self._headers, json=body, timeout=self._timeout)
        
        # Handle 409 Conflict - analyzer already exists, which is fine for updates
        if resp.status_code == 409:
            self._logger.info(f"Analyzer '{analyzer_id}' already exists, update completed")
            return resp
        
        # Log error details for debugging
        if not resp.ok:
            try:
                error_details = resp.json()
                self._logger.error(f"API Error Response: {json.dumps(error_details, indent=2)}")
            except:
                self._logger.error(f"API Error Response Text: {resp.text}")
        
        resp.raise_for_status()
        return resp

    def delete_analyzer(self, analyzer_id: str):
        """Delete an existing custom analyzer.
        
        Args:
            analyzer_id: ID of the analyzer to delete.
            
        Returns:
            HTTP response object with deletion status.
        """
        self._refresh_headers()
        url = f"{self._endpoint}/contentunderstanding/analyzers/{analyzer_id}?api-version={self._api_version}"
        resp = self._session.delete(url, headers=self._headers, timeout=self._timeout)
        resp.raise_for_status()
        return resp

    def classify_document(self, classifier_id: str, file_location: str):
        """Classify a document to identify content type and segments.
        
        Supports both local files (binary upload) and remote URLs.
        Polls operation until completion and returns classification results.
        
        Args:
            classifier_id: ID of classifier to use for categorization.
            file_location: Local file path or HTTP(S) URL to document.
            
        Returns:
            Dict containing classification results with identified categories.
            
        Raises:
            ValueError: If file_location format is invalid or operation header missing.
            FileNotFoundError: If local file doesn't exist.
            requests.HTTPError: If API request fails.
        """
        self._refresh_headers()
        if Path(file_location).exists():
            with open(file_location, "rb") as file:
                data = file.read()
            headers = {"Content-Type": "application/octet-stream"}
            url = f"{self._endpoint}/contentunderstanding/classifiers/{classifier_id}:classify?_overload=classifyBinary&api-version={self._api_version}"
            response = self._session.post(url, headers={**self._headers, **headers}, data=data, timeout=self._timeout)
        elif file_location.startswith(("http://", "https://")):
            data = {"url": file_location}
            headers = {"Content-Type": "application/json"}
            url = f"{self._endpoint}/contentunderstanding/classifiers/{classifier_id}:classify?api-version={self._api_version}"
            response = self._session.post(url, headers={**self._headers, **headers}, json=data, timeout=self._timeout)
        else:
            raise ValueError("File location must be a valid path or URL")
        
        response.raise_for_status()
        operation_location = response.headers.get("operation-location", "")
        if not operation_location:
            raise ValueError("Operation location not found in response headers")
        
        return self._poll_operation_result(operation_location)

    def begin_analyze(self, analyzer_id: str, file_location: str):
        """Extract structured data from document using custom analyzer.
        
        Performs field extraction based on analyzer's schema definition.
        Supports both local files (binary upload) and remote URLs.
        Polls operation until completion and returns extracted data.
        
        Args:
            analyzer_id: ID of custom analyzer with field extraction schema.
            file_location: Local file path or HTTP(S) URL to document.
            
        Returns:
            Dict containing extracted fields with values and confidence scores.
            
        Raises:
            ValueError: If file_location format is invalid or operation header missing.
            FileNotFoundError: If local file doesn't exist.
            requests.HTTPError: If API request fails.
        """
        self._refresh_headers()
        if Path(file_location).exists():
            with open(file_location, "rb") as file:
                data = file.read()
            headers = {"Content-Type": "application/octet-stream"}
            url = f"{self._endpoint}/contentunderstanding/analyzers/{analyzer_id}:analyze?api-version={self._api_version}"
            response = self._session.post(url, headers={**self._headers, **headers}, data=data, timeout=self._timeout)
        elif file_location.startswith(("http://", "https://")):
            data = {"url": file_location}
            headers = {"Content-Type": "application/json"}
            url = f"{self._endpoint}/contentunderstanding/analyzers/{analyzer_id}:analyze?api-version={self._api_version}"
            response = self._session.post(url, headers={**self._headers, **headers}, json=data, timeout=self._timeout)
        else:
            raise ValueError("File location must be a valid path or URL")
        
        response.raise_for_status()
        operation_location = response.headers.get("operation-location", "")
        if not operation_location:
            raise ValueError("Operation location not found in response headers")
        
        return self._poll_operation_result(operation_location)

    def _poll_operation_result(self, operation_location: str, timeout: int = 600):
        """Poll asynchronous operation status until completion.
        
        Implements exponential backoff polling with 2-second intervals.
        Handles succeeded, failed, and timeout scenarios.
        
        Args:
            operation_location: Operation status URL from response header.
            timeout: Maximum seconds to wait before timeout. Defaults to 600 (10 min).
            
        Returns:
            Dict containing operation result data on success.
            
        Raises:
            TimeoutError: If operation doesn't complete within timeout period.
            RuntimeError: If operation status is 'failed'.
            requests.HTTPError: If status polling request fails.
        """
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds")
            
            response = self._session.get(operation_location, headers=self._headers, timeout=self._timeout)
            response.raise_for_status()
            result = response.json()
            status = result.get("status", "").lower()
            
            if status == "succeeded":
                return result.get("result", {})
            elif status == "failed":
                raise RuntimeError("Operation failed")
            
            time.sleep(2)

# --- CLI Menu -----------------------------------------------------------------------
def cli_menu(client: AzureContentUnderstandingClient, settings: Settings):
    """Interactive CLI for managing Content Understanding resources.
    
    Provides a menu-driven interface for classifier and analyzer management,
    document processing, and full extraction workflows.
    
    Args:
        client: Content Understanding API client.
        settings: Configuration including file paths and endpoints.
    """
    def display_items(items: list, item_type: str):
        """Display resource items in a formatted ASCII table.
        
        Handles multiple ID field formats (id, analyzerId, classifierId, name).
        Shows ID, status, and description columns with proper alignment.
        
        Args:
            items: List of resource dictionaries to display.
            item_type: Resource type name for display headers (e.g., 'classifiers').
        """
        if not items:
            print(f"No {item_type} found.")
            return
        
        print(f"\n=== Available {item_type.title()} ===")
        print(f"{'ID':<30} {'Status':<15} {'Description'}")
        print("-" * 80)
        for item in items:
            # Try different possible key names for ID
            item_id = item.get("id") or item.get("analyzerId") or item.get("classifierId") or item.get("name") or "N/A"
            status = item.get("status", "N/A")
            description = item.get("description", "N/A")
            print(f"{item_id:<30} {status:<15} {description}")
        print()

    def select_from_list(items: list, item_type: str) -> str:
        """Interactive prompt for selecting a resource from displayed list.
        
        Shows available items and validates user selection against all possible
        ID field formats. Supports 'back' command to cancel selection.
        
        Args:
            items: List of available resource dictionaries.
            item_type: Resource type name for prompt text.
            
        Returns:
            Selected resource ID string, or empty string if cancelled/no items.
        """
        if not items:
            print(f"No {item_type} available.")
            return ""
        
        display_items(items, item_type)
        while True:
            choice = input(f"Enter {item_type} ID (or 'back' to return): ").strip()
            if choice.lower() == 'back':
                return ""
            # Check against all possible ID field names
            if any((item.get("id") or item.get("analyzerId") or item.get("classifierId") or item.get("name")) == choice for item in items):
                return choice
            print(f"Invalid {item_type} ID. Please try again.")

    def prompt():
        print("\n" + "="*60)
        print("    AZURE CONTENT UNDERSTANDING CLI MENU")
        print("="*60)
        print("1) List all classifiers")
        print("2) List all analyzers") 
        print("3) Create/Update classifier")
        print("4) Delete classifier")
        print("5) Create/Update analyzer")
        print("6) Delete analyzer")
        print("7) Select and run classification")
        print("8) Select and run analysis")
        print("9) Run full extraction workflow")
        print("0) Exit")
        print("-" * 60)
        return input("Choose an option: ").strip()

    while True:
        choice = prompt()
        try:
            if choice == "1":
                classifiers = client.list_classifiers()
                display_items(classifiers, "classifiers")
                
            elif choice == "2":
                analyzers = client.list_analyzers()
                display_items(analyzers, "analyzers")
                
            elif choice == "3":
                cid = input("Classifier ID: ").strip()
                if not cid:
                    print("Classifier ID required.")
                    continue
                analyzers = client.list_analyzers()
                aid = select_from_list(analyzers, "analyzer")
                if aid:
                    client.create_or_update_classifier(cid, aid)
                    print(f"✓ Classifier '{cid}' created/updated successfully")
                    
            elif choice == "4":
                classifiers = client.list_classifiers()
                cid = select_from_list(classifiers, "classifier")
                if cid:
                    confirm = input(f"Delete classifier '{cid}'? (yes/no): ").strip().lower()
                    if confirm == 'yes':
                        client.delete_classifier(cid)
                        print(f"✓ Classifier '{cid}' deleted")
                        
            elif choice == "5":
                aid = input("Analyzer ID: ").strip()
                if not aid:
                    print("Analyzer ID required.")
                    continue
                schema = load_extraction_schema(settings.extraction_schema_path)
                client.create_or_update_analyzer(aid, schema)
                print(f"✓ Analyzer '{aid}' created/updated successfully")
                
            elif choice == "6":
                analyzers = client.list_analyzers()
                aid = select_from_list(analyzers, "analyzer")
                if aid:
                    confirm = input(f"Delete analyzer '{aid}'? (yes/no): ").strip().lower()
                    if confirm == 'yes':
                        client.delete_analyzer(aid)
                        print(f"✓ Analyzer '{aid}' deleted")
                        
            elif choice == "7":
                classifiers = client.list_classifiers()
                cid = select_from_list(classifiers, "classifier")
                if cid:
                    print(f"Running classification with '{cid}'...")
                    result = client.classify_document(cid, settings.file_location)
                    print("Classification Result:")
                    print(json.dumps(result, indent=2))
                    
            elif choice == "8":
                analyzers = client.list_analyzers()
                aid = select_from_list(analyzers, "analyzer")
                if aid:
                    print(f"Running analysis with '{aid}'...")
                    result = client.begin_analyze(aid, settings.file_location)
                    print("Analysis Result:")
                    print(json.dumps(result, indent=2))
                    
            elif choice == "9":
                classifiers = client.list_classifiers()
                analyzers = client.list_analyzers()
                cid = select_from_list(classifiers, "classifier")
                if not cid:
                    continue
                aid = select_from_list(analyzers, "analyzer")
                if not aid:
                    continue
                settings.classifier_id = cid
                settings.analyzer_id = aid
                run_extraction(client, settings)
                
            elif choice == "0":
                print("Exiting...")
                return
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)


# --- Markdown Generation -----------------------------------------------------------
def generate_markdown_summary(analysis_result: dict[str, Any], schema_path: str = "extraction_schema.json") -> str:
    """Generate structured markdown report from extracted invoice data.
    
    Organizes extraction results into sections: general fields (single values),
    object fields (flattened), and array fields (tabular format). Uses schema
    to identify expected fields even if not found in results.
    
    Args:
        analysis_result: API response containing extracted field values.
        schema_path: Path to schema JSON for field type information.
        
    Returns:
        Path to generated markdown file ('invoice_extracted_fields_summary.md').
    """
    try:
        schema = load_extraction_schema(schema_path)
        field_definitions = schema.get("fields", {})
    except Exception as e:
        logging.warning(f"Could not load extraction schema: {e}")
        field_definitions = {}
    
    output_lines = ["# Extracted Invoice Fields\n"]
    
    if "contents" not in analysis_result or not analysis_result["contents"]:
        output_lines.append("No content extracted from document.\n")
        output_path = "invoice_extracted_fields_summary.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(output_lines)
        return output_path
    
    fields = analysis_result["contents"][0].get("fields", {})
    
    # Separate fields into categories
    general_fields = {}
    array_fields = {}
    object_fields = {}
    
    for field_name, field_data in fields.items():
        if not isinstance(field_data, dict):
            continue
            
        # Check if it's an array field
        if "valueArray" in field_data:
            array_fields[field_name] = field_data["valueArray"]
        # Check if it's an object field with sub-properties
        elif "valueObject" in field_data:
            object_fields[field_name] = field_data["valueObject"]
        else:
            # It's a simple field - extract the value
            value = extract_field_value(field_data)
            if value is not None:
                general_fields[field_name] = value
    
    # Also check for fields defined in schema but not in results
    for field_name, field_def in field_definitions.items():
        if field_name not in fields:
            if field_def.get("type") == "array":
                array_fields[field_name] = []
            elif field_def.get("type") == "object":
                object_fields[field_name] = {}
    
    # Write general fields section
    output_lines.append("## General fields\n\n")
    output_lines.append("| field | value |\n")
    output_lines.append("|---|---|\n")
    
    # Process general fields and object sub-fields
    for field_name in sorted(general_fields.keys()):
        value = general_fields[field_name]
        output_lines.append(f"| {field_name} | {format_value(value)} |\n")
    
    # Process object fields (flatten them)
    for obj_name in sorted(object_fields.keys()):
        obj_data = object_fields[obj_name]
        if isinstance(obj_data, dict):
            for sub_field, sub_value in obj_data.items():
                field_value = extract_field_value(sub_value) if isinstance(sub_value, dict) else sub_value
                if field_value is not None:
                    output_lines.append(f"| {obj_name}.{sub_field} | {format_value(field_value)} |\n")
    
    output_lines.append("\n")
    
    # Write array field sections
    for array_name in sorted(array_fields.keys()):
        array_data = array_fields[array_name]
        if not array_data:
            continue
            
        output_lines.append(f"## {array_name}\n\n")
        
        # Determine columns from first item or schema
        columns = []
        if array_data and isinstance(array_data[0], dict):
            # Get fields from first item
            first_item = array_data[0].get("valueObject", array_data[0])
            columns = list(first_item.keys())
        elif array_name in field_definitions and "items" in field_definitions[array_name]:
            # Get fields from schema
            item_props = field_definitions[array_name]["items"].get("properties", {})
            columns = list(item_props.keys())
        
        if columns:
            # Create header row
            header = "| " + " | ".join(f"{array_name}.{col}" for col in columns) + " |\n"
            output_lines.append(header)
            output_lines.append("|" + "---|" * len(columns) + "\n")
            
            # Add data rows
            for item in array_data:
                if isinstance(item, dict):
                    item_obj = item.get("valueObject", item)
                    row_values = []
                    for col in columns:
                        if col in item_obj:
                            val = extract_field_value(item_obj[col]) if isinstance(item_obj[col], dict) else item_obj[col]
                            row_values.append(format_value(val))
                        else:
                            row_values.append("na")
                    output_lines.append("| " + " | ".join(row_values) + " |\n")
        
        output_lines.append("\n\n")
    
    # Write to file
    output_path = "invoice_extracted_fields_summary.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)
    
    return output_path


def extract_field_value(field_data: dict[str, Any]) -> Any:
    """Extract typed value from Content Understanding field response structure.
    
    Checks multiple value type keys in priority order, handling strings,
    numbers, dates, times, phone numbers, booleans, and currency amounts.
    
    Args:
        field_data: Field dictionary with type-specific value keys.
        
    Returns:
        Extracted value of appropriate type, or None if no value found.
    """
    # Check for different value types in order of preference
    for value_key in ["valueString", "valueNumber", "valueDate", "valueTime", 
                      "valuePhoneNumber", "valueInteger", "valueBoolean"]:
        if value_key in field_data:
            return field_data[value_key]
    
    # Check for currency values
    if "valueCurrency" in field_data:
        currency_data = field_data["valueCurrency"]
        if isinstance(currency_data, dict) and "amount" in currency_data:
            return currency_data["amount"]
    
    return None


def format_value(value: Any) -> str:
    """Format extracted values for clean markdown table display.
    
    Converts None/empty to 'na', booleans to lowercase strings,
    and removes unnecessary decimal points from whole numbers.
    
    Args:
        value: Extracted field value of any type.
        
    Returns:
        Formatted string suitable for markdown tables.
    """
    if value is None or value == "":
        return "na"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        # Format numbers consistently
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


# --- Extraction Workflow ------------------------------------------------------------
def run_extraction(client: AzureContentUnderstandingClient, settings: Settings):
    """Execute end-to-end invoice extraction workflow.
    
    Workflow steps:
    1. Validate file accessibility (local or remote)
    2. Classify document to identify invoice segments
    3. Extract fields using custom analyzer if invoice found
    4. Save raw JSON results and generate markdown summary
    5. Display key field values (VendorName, InvoiceId, etc.)
    
    Args:
        client: Configured API client instance.
        settings: Configuration with file paths and resource IDs.
        
    Raises:
        ValueError: If file inaccessible or classifier/analyzer IDs missing.
        Exception: Re-raises any API operation failures.
    """
    loc = settings.file_location
    
    # Validate file location
    if Path(loc).exists():
        print(f"Processing local file: {loc}")
    elif loc.startswith(("http://", "https://")):
        print(f"Processing remote file: {loc}")
        try:
            response = client._session.head(loc, timeout=settings.http_timeout)
            response.raise_for_status()
        except Exception as e:
            raise ValueError(f"Cannot access remote file {loc}: {e}")
    else:
        raise ValueError(f"Invalid file location: {loc}")

    if not settings.classifier_id or not settings.analyzer_id:
        raise ValueError("Both classifier_id and analyzer_id must be set")

    try:
        print(f"\n🔍 Step 1: Classifying document with '{settings.classifier_id}'...")
        classification_result = client.classify_document(settings.classifier_id, loc)
        
        # Check if document has invoice content
        contents = classification_result.get("contents", [])
        invoice_segments = [c for c in contents if c.get("category") == "invoice"]
        
        if not invoice_segments:
            print("❌ Document is not classified as an invoice. Skipping extraction.")
            return
        
        print(f"✅ Found {len(invoice_segments)} invoice segment(s)")
        
        print(f"\n📊 Step 2: Analyzing document with '{settings.analyzer_id}'...")
        analysis_result = client.begin_analyze(settings.analyzer_id, loc)
        
        # Save raw results
        raw_output_file = "invoice_raw_result.json"
        with open(raw_output_file, "w", encoding="utf-8") as f:
            json.dump({
                "classification": classification_result,
                "analysis": analysis_result
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Raw results saved to: {raw_output_file}")
        
        # Extract and display key fields
        if "contents" in analysis_result and analysis_result["contents"]:
            fields = analysis_result["contents"][0].get("fields", {})
            
            print(f"\n📋 Extracted Fields Summary:")
            print("-" * 40)
            
            # Display a few key fields if available
            key_fields = ["VendorName", "InvoiceId", "InvoiceDate", "InvoiceTotal", "CustomerName"]
            for field_name in key_fields:
                if field_name in fields:
                    field_data = fields[field_name]
                    value = "N/A"
                    if isinstance(field_data, dict):
                        # Extract value based on type
                        for val_key in ["valueString", "valueNumber", "valueDate"]:
                            if val_key in field_data:
                                value = field_data[val_key]
                                break
                    print(f"{field_name}: {value}")
            
            print(f"\n📁 Complete analysis saved to: {raw_output_file}")
            
            # Generate markdown summary
            markdown_file = generate_markdown_summary(analysis_result, settings.extraction_schema_path)
            print(f"📄 Markdown summary saved to: {markdown_file}")
            
            print("✅ Extraction workflow completed successfully!")
        else:
            print("⚠️  No field data found in analysis result")
            
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        raise


# --- Main -------------------------------------------------------------------------------
def main():
    """Application entry point with CLI menu or direct extraction modes.
    
    Configures logging, validates environment, and launches either:
    - Interactive CLI menu (default): Full resource management interface
    - Direct extraction (--run flag): Automated workflow execution
    
    Environment Variables:
        CU_ENDPOINT: Azure Content Understanding endpoint (required)
        CU_KEY: Subscription key for auth (or use CU_AAD_TOKEN)
        CU_AAD_TOKEN: AAD bearer token for auth (or use CU_KEY)
        INVOICE_FILE: Document path/URL (default: 'invoice.pdf')
        ANALYZER_ID: Custom analyzer ID (required for --run mode)
        CLASSIFIER_ID: Classifier ID (auto-generated if not set)
        EXTRACTION_SCHEMA_PATH: Schema JSON path (default: 'extraction_schema.json')
    
    Exit Codes:
        0: Success
        1: Configuration error or extraction failure
    """
    parser = argparse.ArgumentParser(description="Azure Content Understanding Invoice Extractor")
    parser.add_argument("--run", action="store_true", 
                       help="Run extraction directly without CLI menu (requires ANALYZER_ID env var)")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s │ %(message)s")
    endpoint = os.environ.get("CU_ENDPOINT") or ""
    key = os.environ.get("CU_KEY")
    aad = os.environ.get("CU_AAD_TOKEN")
    analyzer_id = os.environ.get("ANALYZER_ID", "")
    classifier_id = os.environ.get("CLASSIFIER_ID", "")
    
    settings = Settings(
        endpoint=endpoint,
        subscription_key=key,
        aad_token=aad,
        analyzer_id=analyzer_id,
        classifier_id=classifier_id,
        file_location=os.environ.get("INVOICE_FILE", "invoice.pdf"),
    )
    
    client = AzureContentUnderstandingClient(
        settings.endpoint,
        settings.api_version,
        subscription_key=settings.subscription_key,
        token_provider=settings.token_provider,
        http_timeout=settings.http_timeout,
        retries=settings.retries,
        backoff_factor=settings.backoff_factor,
    )
    
    if args.run:
        if not analyzer_id:
            print("Error: ANALYZER_ID environment variable is required when using --run flag", file=sys.stderr)
            sys.exit(1)
        
        # Set up classifier and analyzer for direct extraction
        if not classifier_id:
            # Create a default classifier ID if not provided
            classifier_id = f"{analyzer_id}-classifier"
            settings.classifier_id = classifier_id
            print(f"No CLASSIFIER_ID provided, using default: {classifier_id}")
        
        try:
            # Load extraction schema and set up resources
            print(f"Loading extraction schema from: {settings.extraction_schema_path}")
            schema = load_extraction_schema(settings.extraction_schema_path)
            
            print(f"Creating/updating analyzer: {analyzer_id}")
            client.create_or_update_analyzer(analyzer_id, schema)
            
            print(f"Creating/updating classifier: {classifier_id}")
            client.create_or_update_classifier(classifier_id, analyzer_id)
            
            # Wait a moment for resources to be ready
            print("Waiting for resources to be ready...")
            time.sleep(3)
            
            # Run the extraction workflow
            run_extraction(client, settings)
            
        except Exception as e:
            print(f"Error during extraction: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        cli_menu(client, settings)


if __name__ == "__main__":
    main()
