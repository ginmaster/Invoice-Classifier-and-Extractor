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
    """Load extraction schema from JSON file.
    
    Args:
        path: Path to the extraction schema JSON file.
        
    Returns:
        The extraction schema dictionary.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
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
        """Get AAD token provider if AAD authentication is configured.
        
        Returns:
            Token provider callable or None if using subscription key.
        """
        if self.aad_token:
            return AADTokenProvider(self.aad_token)
        return None

# --- Token Provider ---------------------------------------------------------------
class AADTokenProvider:
    """Azure Active Directory token provider with expiration handling.
    
    Args:
        initial_token: The AAD token to use.
        expires_in: Token lifetime in seconds.
    """
    def __init__(self, initial_token: str, expires_in: int = 3300):
        self._token = initial_token
        self._expires_at = time.time() + expires_in

    def __call__(self) -> str:
        """Get current token, refreshing if near expiration.
        
        Returns:
            Current valid AAD token.
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
        """Build HTTP headers with authentication.
        
        Returns:
            Dictionary of HTTP headers.
        """
        token = self._token_provider()() if self._token_provider else None
        headers = {"x-ms-useragent": "invoice-extractor"}
        if self._subscription_key:
            headers["Ocp-Apim-Subscription-Key"] = self._subscription_key
        elif token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry configuration.
        
        Returns:
            Configured requests session.
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
        """Refresh authentication headers."""
        self._headers = self._get_headers()

    # Classifier operations
    def list_classifiers(self) -> list[dict[str, Any]]:
        """List all document classifiers.
        
        Returns:
            List of classifier definitions.
        """
        self._refresh_headers()
        url = f"{self._endpoint}/contentunderstanding/classifiers?api-version={self._api_version}"
        resp = self._session.get(url, headers=self._headers, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json().get("value", [])

    def create_or_update_classifier(self, classifier_id: str, analyzer_id: str):
        """Create or update a document classifier.
        
        Args:
            classifier_id: Unique identifier for the classifier.
            analyzer_id: Associated analyzer for invoice processing.
            
        Returns:
            HTTP response object.
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
        """Delete a document classifier.
        
        Args:
            classifier_id: Classifier to delete.
            
        Returns:
            HTTP response object.
        """
        self._refresh_headers()
        url = f"{self._endpoint}/contentunderstanding/classifiers/{classifier_id}?api-version={self._api_version}"
        resp = self._session.delete(url, headers=self._headers, timeout=self._timeout)
        resp.raise_for_status()
        return resp

    # Analyzer operations
    def list_analyzers(self) -> list[dict[str, Any]]:
        """List all custom analyzers.
        
        Returns:
            List of analyzer definitions.
        """
        self._refresh_headers()
        url = f"{self._endpoint}/contentunderstanding/analyzers?api-version={self._api_version}"
        resp = self._session.get(url, headers=self._headers, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json().get("value", [])

    def create_or_update_analyzer(self, analyzer_id: str, field_schema: dict[str, Any], description: str = "Custom Invoice Analyzer"):
        """Create or update a custom field extraction analyzer.
        
        Args:
            analyzer_id: Unique identifier for the analyzer.
            field_schema: Field extraction schema definition.
            description: Human-readable description.
            
        Returns:
            HTTP response object.
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
        
        resp.raise_for_status()
        return resp

    def delete_analyzer(self, analyzer_id: str):
        """Delete a custom analyzer.
        
        Args:
            analyzer_id: Analyzer to delete.
            
        Returns:
            HTTP response object.
        """
        self._refresh_headers()
        url = f"{self._endpoint}/contentunderstanding/analyzers/{analyzer_id}?api-version={self._api_version}"
        resp = self._session.delete(url, headers=self._headers, timeout=self._timeout)
        resp.raise_for_status()
        return resp

    def classify_document(self, classifier_id: str, file_location: str):
        """Classify document content using specified classifier.
        
        Args:
            classifier_id: Classifier to use for document classification.
            file_location: Path to local file or URL to remote document.
            
        Returns:
            Classification results with identified content categories.
            
        Raises:
            ValueError: If file location is invalid or operation location missing.
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
        """Extract structured fields from document using custom analyzer.
        
        Args:
            analyzer_id: Custom analyzer for field extraction.
            file_location: Path to local file or URL to remote document.
            
        Returns:
            Analysis results with extracted field values.
            
        Raises:
            ValueError: If file location is invalid or operation location missing.
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
        """Poll long-running operation until completion.
        
        Args:
            operation_location: URL to poll for operation status.
            timeout: Maximum time to wait in seconds.
            
        Returns:
            Operation result data.
            
        Raises:
            TimeoutError: If operation exceeds timeout.
            RuntimeError: If operation fails.
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
        """Display items in formatted table.
        
        Args:
            items: List of items to display.
            item_type: Type description for headers.
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
        """Allow user to select item from list.
        
        Args:
            items: Available items to choose from.
            item_type: Type description for prompts.
            
        Returns:
            Selected item ID or empty string if cancelled.
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


# --- Extraction Workflow ------------------------------------------------------------
def run_extraction(client: AzureContentUnderstandingClient, settings: Settings):
    """Execute complete invoice extraction workflow.
    
    Performs document classification, field extraction, and result processing.
    Saves raw results to JSON and displays key extracted fields.
    
    Args:
        client: Content Understanding API client.
        settings: Configuration with file location and analyzer/classifier IDs.
        
    Raises:
        ValueError: If file location is invalid or required settings missing.
        Exception: If classification or analysis operations fail.
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
            print("✅ Extraction workflow completed successfully!")
        else:
            print("⚠️  No field data found in analysis result")
            
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        raise


# --- Main -------------------------------------------------------------------------------
def main():
    """Initialize application and either launch CLI menu or run extraction directly.
    
    Configures logging, loads environment variables, and starts the
    interactive Content Understanding CLI or runs extraction workflow directly.
    
    Environment Variables:
        CU_ENDPOINT: Content Understanding service endpoint URL.
        CU_KEY: Subscription key for authentication.
        CU_AAD_TOKEN: AAD token for authentication (alternative to CU_KEY).
        INVOICE_FILE: Path to invoice file to process.
        ANALYZER_ID: Custom analyzer identifier (required for --run).
        CLASSIFIER_ID: Document classifier identifier (optional for --run).
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
