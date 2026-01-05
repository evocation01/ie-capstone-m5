import base64
import io
import zlib
from pathlib import Path

import requests
from PIL import Image


def generate_fishbone_diagram(output_filename="inventory_cost_fishbone.png"):
    """
    Generates a fishbone (Ishikawa) diagram for Inventory Cost root causes
    using Mermaid syntax and the Kroki.io service.
    """
    # 1. Define the Fishbone Diagram in Mermaid Syntax
    # 1. Define the Diagram in a supported Mermaid Syntax (Graph LR for a more compact layout)
    graph_definition = """
    %%{init: {'theme': 'default', 'themeVariables': { 'primaryColor': '#E0E7FF', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#fff'}}}%%
    graph LR
        subgraph "Cause:&nbsp;Forecast&nbsp;Error"
            A1["Model Inaccuracy (RMSE)"] --> E{Forecast Error}
            A2["Sparsity Penalty Effect"] --> E
            A3["Incorrect Forecast Horizon"] --> E
        end

        subgraph "Cause:&nbsp;Demand&nbsp;Uncertainty"
            B1["High Seasonality"] --> F{Demand Uncertainty}
            B2["Unaccounted Promotions"] --> F
            B3["Random Market Volatility"] --> F
        end

        subgraph "Cause:&nbsp;Lead&nbsp;Time&nbsp;Variability"
            C1["Supplier Production Delays"] --> G{Lead Time Variability}
            C2["Logistics & Shipping Issues"] --> G
            C3["Quality Inspection Holds"] --> G
        end

        subgraph "Cause:&nbsp;Parameter&nbsp;Miscalibration"
            D1["Incorrect Service Level"] --> H{Parameter Miscalibration}
            D2["Inaccurate Holding Costs"] --> H
            D3["Inaccurate Stockout Costs"] --> H
        end

        E & F & G & H --> Z((Inventory Cost));

        classDef cause fill:#fff5e6,stroke:#f5a623,stroke-width:2px,color:#333;
        class E,F,G,H cause;
        classDef effect fill:#ffeef0,stroke:#d0021b,stroke-width:2px,color:#333;
        class Z effect;
    """

    # 2. Compress and Encode for URL (same as other script)
    data_bytes = graph_definition.encode("utf-8")
    compressed_data = zlib.compress(data_bytes, level=9)
    base64_bytes = base64.urlsafe_b64encode(compressed_data)
    base64_string = base64_bytes.decode("ascii")

    # 3. Fetch from Kroki.io
    url = f"https://kroki.io/mermaid/png/{base64_string}"
    print("Fetching fishbone diagram from Kroki.io...")

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            # Open the image data from the API response
            img_data = io.BytesIO(response.content)
            original_image = Image.open(img_data).convert("RGBA")

            # 4. Force a white background for consistency
            white_bg = Image.new("RGBA", original_image.size, "WHITE")
            white_bg.paste(original_image, (0, 0), original_image)

            # Define output directory and save the file
            output_dir = Path("backend/reports/diagrams/img/diagrams")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / output_filename
            
            white_bg.save(output_file)
            print(f"✅ Success! Diagram saved to: {output_file}")
        else:
            print(f"❌ API Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Network or Python Error: {e}")


if __name__ == "__main__":
    generate_fishbone_diagram()
