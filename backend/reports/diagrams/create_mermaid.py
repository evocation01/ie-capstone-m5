import base64
import io
import zlib

import requests
from PIL import Image


def generate_white_diagram(output_filename="pipeline_diagram_white.png"):
    # 1. Define the Graph with "default" (light) theme styling
    graph_definition = """
    %%{init: {'theme': 'default', 'themeVariables': { 'primaryColor': '#E0E7FF', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#fff'}}}%%
    graph TD
        classDef source fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#333;
        classDef process fill:#e1efff,stroke:#4a90e2,stroke-width:2px,color:#333,rx:5,ry:5;
        classDef decision fill:#fff5e6,stroke:#f5a623,stroke-width:2px,color:#333,rx:5,ry:5;
        classDef error fill:#ffeef0,stroke:#d0021b,stroke-width:2px,color:#333;
        classDef ai fill:#e8fcf5,stroke:#00b894,stroke-width:2px,color:#333;
        classDef final fill:#333,stroke:#333,stroke-width:2px,color:#fff;

        A[Raw Data Sources]:::source -->|Ingest| B(Data Processing Pipeline):::process
        B --> C{Data Quality Checks}:::decision
        
        C -- Fail --> D[Log Error / Cleaning]:::error
        C -- Pass --> E[Feature Engineering]:::process
        
        E --> F[Classical Modeling Branch]:::process
        E --> G[AI/Deep Learning Branch]:::ai
        
        F --> H[Baseline Forecast]:::process
        G --> I[Advanced Forecast]:::ai
        
        H --> J[Optimization Engine]:::process
        I --> J
        
        J --> K[Cost Comparison & Decision Support]:::final
    """

    # 2. Compress and Encode (Solves the '204' URL length error)
    data_bytes = graph_definition.encode("utf-8")
    compressed_data = zlib.compress(data_bytes, level=9)
    base64_bytes = base64.urlsafe_b64encode(compressed_data)
    base64_string = base64_bytes.decode("ascii")

    # 3. Fetch from Kroki (High reliability)
    url = f"https://kroki.io/mermaid/png/{base64_string}"

    print("Fetching diagram...")

    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Open the image from the API
            img_data = io.BytesIO(response.content)
            original_image = Image.open(img_data).convert("RGBA")

            # 4. FORCE WHITE BACKGROUND
            # Create a white canvas of the same size
            white_bg = Image.new("RGBA", original_image.size, "WHITE")
            # Paste the diagram on top (using alpha channel as mask)
            white_bg.paste(original_image, (0, 0), original_image)

            # Save as PNG
            output_dir = Path("backend/reports/diagrams/img")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / output_filename
            white_bg.save(output_file)
            print(f"✅ Success! Saved 'White Mode' diagram to: {output_file}")
        else:
            print(f"❌ API Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Python Error: {e}")


if __name__ == "__main__":
    generate_white_diagram()
