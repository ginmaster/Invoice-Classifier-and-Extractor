# Invoice Extractor - Azure Content Understanding Tool

A comprehensive Python CLI tool for extracting structured data from invoices using Azure Content Understanding APIs. This tool provides automated document classification, custom field extraction, and a complete workflow for processing invoice documents.

## Features

### 🔍 Core Capabilities
- **Document Classification**: Automatically classify documents as invoices vs receipts
- **Custom Field Extraction**: Extract 30+ structured fields from invoices including:
  - Invoice header data (number, dates, amounts)
  - Vendor and customer information  
  - Line items with detailed breakdowns
  - VAT/tax calculations
  - Payment terms and banking details
  - Prepayment tracking
- **Multi-format Support**: Process local PDF files or remote URLs
- **Robust Error Handling**: Exponential backoff, timeout protection, and retry logic
- **Interactive CLI**: User-friendly menu system for managing analyzers and classifiers

### 🛠️ Technical Features
- Session-based HTTP client with automatic retries
- AAD token and subscription key authentication
- Configurable timeouts and retry policies
- Comprehensive logging and error reporting
- Type-safe configuration with dataclasses

## Installation

### Prerequisites
- Python 3.8+
- Azure Content Understanding resource
- Valid Azure subscription

### Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install requests
```

For lab exercises, you can also use:
```bash
pip install -r mslearn-ai-doc/Labfiles/05-content-understanding/code/requirements.txt
```

## Configuration

### Required Environment Variables
```bash
export CU_ENDPOINT="https://<your-resource>.services.ai.azure.com"
export CU_KEY="your-subscription-key"  # OR use CU_AAD_TOKEN for AAD auth
```

### Optional Environment Variables
```bash
export ANALYZER_ID="invoice-v1"                    # Custom analyzer ID
export CLASSIFIER_ID="invoice-classifier"          # Document classifier ID  
export INVOICE_FILE="path/to/invoice.pdf"         # Invoice file path or URL
export EXTRACTION_SCHEMA_PATH="schema.json"       # Custom schema file
```

## Usage

### Quick Start
```bash
python invoiceExtract.py
```

This launches an interactive CLI menu with the following options:

### CLI Menu Options
1. **List all classifiers** - View existing document classifiers
2. **List all analyzers** - View existing custom analyzers
3. **Create/Update classifier** - Set up document classification
4. **Delete classifier** - Remove document classifiers
5. **Create/Update analyzer** - Configure custom field extraction
6. **Delete analyzer** - Remove custom analyzers
7. **Select and run classification** - Classify a single document
8. **Select and run analysis** - Extract fields from a document
9. **Run full extraction workflow** - Complete end-to-end processing
0. **Exit** - Close the application

### Full Extraction Workflow

The complete workflow (option 9) performs:

1. **Document Classification** - Identifies invoice segments in the document
2. **Field Extraction** - Extracts structured data using custom analyzer
3. **Results Processing** - Saves raw JSON and displays key fields
4. **Summary Report** - Shows extracted invoice summary

### Example Output
```
🔍 Step 1: Classifying document with 'invoice-classifier'...
✅ Found 1 invoice segment(s)

📊 Step 2: Analyzing document with 'invoice-v1'...
✅ Raw results saved to: invoice_raw_result.json

📋 Extracted Fields Summary:
----------------------------------------
VendorName: ACME Corporation
InvoiceId: INV-2024-001
InvoiceDate: 2024-01-15
InvoiceTotal: 1250.00
CustomerName: Example Client Ltd

✅ Extraction workflow completed successfully!
```

## Field Schema

The tool extracts comprehensive invoice data including:

### Header Fields
- `no` - Invoice number
- `postingDate` - Invoice date
- `dueDate` - Payment due date
- `amountinclVAT` - Total amount including VAT
- `amountexclVAT` - Net amount excluding VAT
- `vatAmount` - Total VAT amount

### Vendor Information
- `payToName` - Vendor name
- `payToAddressParts` - Structured vendor address
- `vendorVatRegistrationNo` - Vendor VAT ID
- `vendorBank` - Banking details (IBAN, BIC, bank name)

### Customer Information
- `buyFromVendorName` - Customer name
- `buyFromAddress` - Customer address
- `buyFromVendorNo` - Customer reference number
- `customerVatRegistrationNo` - Customer VAT ID

### Line Items
- `items` - Array of invoice line items with:
  - Product/service description
  - Quantities and units of measure
  - Unit prices and line totals
  - VAT rates and amounts
  - Product codes and line numbers

### Payment Terms
- `earlyPaymentDiscountPercentage` - Early payment discount rate
- `earlyPaymentDiscountDate` - Discount deadline
- `netPaymentDueDate` - Net payment due date

### Advanced Features
- `prepaymentInvoices` - Referenced prepayment invoices
- `prepaymentsReceived` - Actual prepayments received
- `shipToName` / `shipToAddress` - Delivery information

## Error Handling

The tool includes comprehensive error handling:

- **Validation**: Endpoint URLs, file paths, and authentication
- **HTTP Retries**: Automatic retry with exponential backoff for transient failures
- **Timeouts**: Configurable timeouts for all operations
- **Resource Conflicts**: Graceful handling of existing analyzers/classifiers
- **Polling**: Long-running operation support with timeout protection

## Authentication

### Subscription Key (Recommended)
```bash
export CU_ENDPOINT="https://your-resource.services.ai.azure.com"
export CU_KEY="your-subscription-key"
```

### Azure Active Directory
```bash
export CU_ENDPOINT="https://your-resource.services.ai.azure.com"  
export CU_AAD_TOKEN="your-aad-token"
```

## File Processing

### Local Files
```bash
export INVOICE_FILE="/path/to/invoice.pdf"
python invoiceExtract.py
```

### Remote URLs
```bash
export INVOICE_FILE="https://example.com/invoice.pdf"
python invoiceExtract.py
```

## Output Files

- `invoice_raw_result.json` - Complete API response with classification and analysis results
- Console output - Formatted summary of key extracted fields
- Logs - Detailed operation logs with timestamps

## Troubleshooting

### Common Issues

**Authentication Errors**
- Verify `CU_ENDPOINT` and `CU_KEY` environment variables
- Check Azure resource permissions and subscription status

**File Not Found**
- Ensure file path is correct and accessible
- For URLs, verify the document is publicly accessible

**Timeout Errors**
- Increase timeout values in configuration
- Check network connectivity to Azure endpoints

**Resource Conflicts (409)**
- Analyzer/classifier already exists (this is handled gracefully)
- Use CLI menu to view existing resources

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Architecture

### Client Design
- `AzureContentUnderstandingClient` - Main API client with session management
- `Settings` - Type-safe configuration using dataclasses
- `AADTokenProvider` - Token refresh handling for AAD authentication

### HTTP Layer
- Session-based requests with connection pooling
- Configurable retry strategy with exponential backoff
- Automatic header refresh for token-based auth

### Processing Pipeline
1. **Validation** - Endpoint, credentials, and file accessibility
2. **Classification** - Document type identification
3. **Analysis** - Field extraction using custom schema
4. **Post-processing** - Result formatting and output generation

## Contributing

To extend the tool:

1. **Add New Fields** - Update `extraction_schema.json` with new field definitions
2. **Custom Processing** - Modify the `run_extraction()` function for custom workflows
3. **Additional Formats** - Extend file handling for new document types
4. **Output Formats** - Add new result formatters in the processing pipeline

## Related Resources

- [Azure Content Understanding Documentation](https://docs.microsoft.com/azure/ai-services/content-understanding/)
- [Microsoft Learn Labs](./mslearn-ai-doc/) - Complete lab exercises and samples
- [Schema Reference](./extraction_schema.json) - Field extraction schema definition
