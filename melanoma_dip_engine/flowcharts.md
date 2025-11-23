# Melanoma DIP Engine - System Flowcharts

This document contains Mermaid flowcharts visualizing the complete melanoma risk prediction system architecture and processing pipelines.

## 1. Overall System Architecture

```mermaid
flowchart TD
    A[Input Dermoscopic Image] --> B[Preprocessing]
    B --> C[CLAHE Contrast Enhancement]
    C --> D[Color Space Conversion<br/>RGB, HSV, CIELab]
    D --> E[Artifact Removal]
    E --> F[Hair Detection<br/>Black Hat Morphology]
    F --> G[TELEA Inpainting]
    G --> H{Parallel Processing}
    
    H --> I[Classification Branch]
    H --> J[Segmentation Branch]
    H --> K[Feature Analysis Branch]
    
    I --> I1[YOLOv11 Model<br/>ISIC 2019 + PH2<br/>10k images]
    I1 --> I2[Binary Classification<br/>Melanoma vs Benign]
    I2 --> I3[Confidence Score<br/>Threshold: 0.35]
    
    J --> J1[Initial Segmentation<br/>Multi-Method Fusion]
    J1 --> J2[Adaptive Thresholding]
    J1 --> J3[HSV Color Segmentation]
    J1 --> J4[Intensity Validation]
    J2 --> J5[Contour Selection<br/>Area + Shape Quality]
    J3 --> J5
    J4 --> J5
    J5 --> J6[GrabCut Refinement<br/>5 Iterations]
    J6 --> J7[Refined Lesion Mask]
    
    K --> K1[ABCD+T Feature Extraction]
    K1 --> K2[Asymmetry<br/>Rotation-Invariant]
    K1 --> K3[Border Irregularity<br/>Compactness + Solidity]
    K1 --> K4[Color Variation<br/>K-Means Clustering]
    K1 --> K5[Diameter<br/>Multiple Methods]
    K1 --> K6[Texture Analysis<br/>GLCM, LBP, Statistical]
    K1 --> K7[Advanced DIP Features]
    K7 --> K8[FFT Frequency Analysis]
    K7 --> K9[Blue-White Veil Detection]
    
    I3 --> L[Risk Scoring Framework]
    J7 --> L
    K2 --> L
    K3 --> L
    K4 --> L
    K5 --> L
    K6 --> L
    K8 --> L
    K9 --> L
    
    L --> M[Normalized Feature Scores<br/>0-1 Scale]
    M --> N[Overall Risk Score<br/>Arithmetic Mean]
    N --> O[Risk Level Categorization<br/>Low/Moderate/High]
    O --> P[Final Assessment<br/>Dual: YOLO + Features]
    P --> Q[Streamlit Web Interface<br/>Interactive Visualization]
    
    style A fill:#e1f5ff
    style P fill:#90EE90
    style Q fill:#FFD700
    style I1 fill:#FFB6C1
    style J7 fill:#87CEEB
    style K1 fill:#DDA0DD
```

## 2. Classification Pipeline (YOLOv11)

```mermaid
flowchart LR
    A[Input Image<br/>512x512] --> B[Image Preprocessing]
    B --> C[Normalization]
    C --> D[YOLOv11 Model<br/>Pretrained on ImageNet]
    D --> E[Fine-Tuned on<br/>ISIC 2019 + PH2<br/>~10,000 images]
    E --> F[Binary Classification Head]
    F --> G{Melanoma<br/>Probability}
    G -->|> 0.35| H[High Risk<br/>Melanoma Detected]
    G -->|≤ 0.35| I[Low Risk<br/>Benign]
    H --> J[Confidence Score<br/>93% Accuracy]
    I --> J
    J --> K[Integration with<br/>Feature-Based Scoring]
    
    style A fill:#e1f5ff
    style E fill:#FFB6C1
    style J fill:#90EE90
    style K fill:#FFD700
```

## 3. Segmentation Pipeline

```mermaid
flowchart TD
    A[Hair-Free Image] --> B[Color Space Conversion]
    B --> C[LAB Color Space]
    B --> D[HSV Color Space]
    B --> E[Grayscale]
    
    C --> F[Method 1:<br/>Adaptive Thresholding<br/>Block Size: 11]
    D --> G[Method 2:<br/>HSV Color Segmentation<br/>Hue: 0-179]
    E --> H[Method 3:<br/>Intensity Validation<br/>55th Percentile]
    
    F --> I{Mask Fusion}
    G --> I
    H --> I
    
    I --> J{Combined Mask<br/>Area Check}
    J -->|> 30% of adaptive| K[Use Combined Mask]
    J -->|≤ 30% of adaptive| L[Use Adaptive Mask Only]
    
    K --> M[Morphological Refinement<br/>2x2 Elliptical Kernel]
    L --> M
    
    M --> N[Contour Detection]
    N --> O[Contour Validation<br/>Area: 100-200k pixels<br/>Aspect Ratio < 10<br/>Solidity > 0.3]
    
    O --> P[Composite Scoring<br/>S = Area × Quality]
    P --> Q[Select Best Contour]
    Q --> R[Initial Segmentation Mask]
    
    R --> S[GrabCut Initialization]
    S --> T[Generate Trimap]
    T --> U[Definite Foreground<br/>Erode 10px]
    T --> V[Definite Background<br/>Dilate 20px]
    T --> W[Probable Foreground<br/>Boundary Region]
    
    U --> X[GrabCut Optimization<br/>5 Iterations<br/>GMM Modeling]
    V --> X
    W --> X
    
    X --> Y{Refined Mask<br/>Validation}
    Y -->|Valid Area| Z[Refined Segmentation<br/>Dice: 0.939]
    Y -->|Invalid Area| R
    
    Z --> AA[Final Lesion Mask<br/>+ Contour]
    
    style A fill:#e1f5ff
    style R fill:#87CEEB
    style Z fill:#90EE90
    style AA fill:#FFD700
```

## 4. Feature Analysis Pipeline (ABCD+T)

```mermaid
flowchart TD
    A[Segmented Lesion<br/>Mask + Contour] --> B[Feature Extraction Module]
    
    B --> C[Asymmetry Analysis]
    C --> C1[Calculate Image Moments]
    C1 --> C2[Rotate to Align Major Axis]
    C2 --> C3[Horizontal Flip Comparison]
    C2 --> C4[Vertical Flip Comparison]
    C3 --> C5[XOR Operation]
    C4 --> C5
    C5 --> C6[Asymmetry Score<br/>0.0 - 1.0]
    
    B --> D[Border Irregularity]
    D --> D1[Calculate Perimeter]
    D --> D2[Calculate Area]
    D1 --> D3[Compactness<br/>C = P²/4πA]
    D2 --> D3
    D --> D4[Convex Hull]
    D4 --> D5[Solidity<br/>S = A/A_hull]
    D3 --> D6[Combined Score<br/>B = 0.5C + 2.0(1-S)]
    D5 --> D6
    
    B --> E[Color Variation]
    E --> E1[Extract Lesion Pixels<br/>RGB Space]
    E1 --> E2[K-Means Clustering<br/>k = 2 to 6]
    E2 --> E3[Validate Cluster Sizes<br/>≥ 2% of pixels]
    E3 --> E4[Color Count<br/>1-6 scale]
    
    B --> F[Diameter Measurement]
    F --> F1[Equivalent Diameter<br/>D_eq = 2√(A/π)]
    F --> F2[Max Feret Diameter<br/>Convex Hull Points]
    F --> F3[Bounding Box Diagonal]
    F1 --> F4[Convert to mm<br/>10 pixels/mm]
    F2 --> F4
    F3 --> F4
    F4 --> F5[Largest Diameter<br/>Clinical Significance]
    
    B --> G[Texture Analysis]
    G --> G1[GLCM Features<br/>Distances: 1,2,3<br/>Angles: 0°,45°,90°,135°]
    G1 --> G2[Contrast, Homogeneity<br/>Energy, Correlation]
    G --> G3[LBP Analysis<br/>Radius: 1, Points: 8]
    G3 --> G4[Uniformity, Contrast<br/>Entropy]
    G --> G5[Statistical Features<br/>Mean, Std, Skewness]
    G --> G6[Gradient Features<br/>Sobel Operators]
    G6 --> G7[Gradient Magnitude<br/>Mean, Std]
    
    B --> H[Advanced DIP Features]
    H --> H1[FFT Frequency Analysis]
    H1 --> H2[2D FFT Transform]
    H2 --> H3[High-Pass Filter<br/>Radius: 10% of image]
    H3 --> H4[High-Frequency Energy<br/>High/Low Ratio]
    H --> H5[Blue-White Veil Detection]
    H5 --> H6[Blue-Green Dominance<br/>B+G > 1.2R]
    H5 --> H7[Blue Ratio<br/>B/R > 1.1]
    H5 --> H8[Localized Enhancement<br/>B > μ_B + 0.5σ_B]
    H5 --> H9[High Luminance<br/>L > 0.5]
    H6 --> H10[Veil Detection<br/>Coverage ≥ 1%]
    H7 --> H10
    H8 --> H10
    H9 --> H10
    
    C6 --> I[Feature Normalization]
    D6 --> I
    E4 --> I
    F5 --> I
    G2 --> I
    G4 --> I
    G7 --> I
    H4 --> I
    H10 --> I
    
    I --> J[Normalized Scores<br/>0-1 Scale]
    J --> K[Risk Score Calculation<br/>R = (A_n + B_n + C_n + T_n)/4]
    K --> L[Risk Level<br/>Low < 0.3<br/>Moderate 0.3-0.7<br/>High ≥ 0.7]
    
    style A fill:#e1f5ff
    style C6 fill:#FFB6C1
    style D6 fill:#87CEEB
    style E4 fill:#DDA0DD
    style F5 fill:#F0E68C
    style G2 fill:#98D8C8
    style H4 fill:#FFA07A
    style H10 fill:#FF69B4
    style L fill:#90EE90
```

## 5. Complete Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant Preprocessing
    participant HairRemoval
    participant Segmentation
    participant YOLO
    participant FeatureExtraction
    participant RiskScoring
    participant Visualization

    User->>Streamlit: Upload Dermoscopic Image
    Streamlit->>Preprocessing: Load Image (512x512)
    Preprocessing->>Preprocessing: CLAHE on L Channel
    Preprocessing->>Preprocessing: Convert to RGB/HSV/LAB
    Preprocessing->>HairRemoval: Preprocessed Image
    
    HairRemoval->>HairRemoval: Black Hat Morphology
    HairRemoval->>HairRemoval: TELEA Inpainting
    HairRemoval->>Segmentation: Hair-Free Image
    
    Segmentation->>Segmentation: Multi-Method Fusion
    Segmentation->>Segmentation: Contour Selection
    Segmentation->>Segmentation: GrabCut Refinement
    Segmentation->>FeatureExtraction: Refined Mask + Contour
    Segmentation->>YOLO: Hair-Free Image
    
    YOLO->>YOLO: Binary Classification
    YOLO->>RiskScoring: Probability Score (93% Acc)
    
    FeatureExtraction->>FeatureExtraction: Extract ABCD+T Features
    FeatureExtraction->>FeatureExtraction: FFT Analysis
    FeatureExtraction->>FeatureExtraction: Blue-White Veil Detection
    FeatureExtraction->>RiskScoring: All Feature Scores
    
    RiskScoring->>RiskScoring: Normalize Features (0-1)
    RiskScoring->>RiskScoring: Calculate Overall Risk
    RiskScoring->>RiskScoring: Categorize Risk Level
    RiskScoring->>Visualization: Risk Assessment
    
    Visualization->>Visualization: Generate Charts
    Visualization->>Visualization: Create Overlays
    Visualization->>Streamlit: Complete Results
    Streamlit->>User: Display Interactive Dashboard
```

## 6. Risk Scoring Framework

```mermaid
flowchart LR
    A[YOLO Classification<br/>93% Accuracy] --> E[Final Risk Assessment]
    B[Asymmetry Score<br/>Normalized] --> F[Feature-Based Risk]
    C[Border Score<br/>Normalized] --> F
    D[Color Score<br/>Normalized] --> F
    G[Texture Score<br/>Normalized] --> F
    
    F --> H[Overall Risk Score<br/>R = Mean of ABCD+T]
    H --> I{Risk Level}
    I -->|R < 0.3| J[LOW RISK<br/>Benign Tendency]
    I -->|0.3 ≤ R < 0.7| K[MODERATE RISK<br/>Requires Monitoring]
    I -->|R ≥ 0.7| L[HIGH RISK<br/>Melanoma Suspected]
    
    A --> M{Dual Assessment}
    J --> M
    K --> M
    L --> M
    
    M --> N[Final Recommendation<br/>Triage Decision]
    
    style A fill:#FFB6C1
    style F fill:#DDA0DD
    style H fill:#87CEEB
    style J fill:#90EE90
    style K fill:#FFD700
    style L fill:#FF6B6B
    style N fill:#FFD700
```

## 7. Data Flow Diagram

```mermaid
flowchart TB
    subgraph Input["Input Stage"]
        I1[Dermoscopic Image<br/>JPG/PNG/BMP]
    end
    
    subgraph Preprocessing["Preprocessing Stage"]
        P1[Resize to 512x512]
        P2[CLAHE Enhancement]
        P3[Color Space Conversion]
    end
    
    subgraph ArtifactRemoval["Artifact Removal"]
        AR1[Hair Detection]
        AR2[TELEA Inpainting]
    end
    
    subgraph Classification["Classification Branch"]
        CL1[YOLOv11 Model]
        CL2[Binary Output]
        CL3[Confidence Score]
    end
    
    subgraph Segmentation["Segmentation Branch"]
        SG1[Multi-Method Fusion]
        SG2[GrabCut Refinement]
        SG3[Final Mask]
    end
    
    subgraph FeatureExtraction["Feature Extraction Branch"]
        FE1[ABCD Features]
        FE2[Texture Analysis]
        FE3[Advanced DIP]
    end
    
    subgraph Scoring["Risk Scoring"]
        SC1[Normalization]
        SC2[Aggregation]
        SC3[Risk Level]
    end
    
    subgraph Output["Output Stage"]
        O1[Interactive Dashboard]
        O2[Visualizations]
        O3[CSV Export]
    end
    
    I1 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> AR1
    AR1 --> AR2
    
    AR2 --> CL1
    AR2 --> SG1
    AR2 --> FE1
    
    CL1 --> CL2
    CL2 --> CL3
    
    SG1 --> SG2
    SG2 --> SG3
    SG3 --> FE1
    
    FE1 --> FE2
    FE2 --> FE3
    
    CL3 --> SC1
    FE3 --> SC1
    SC1 --> SC2
    SC2 --> SC3
    
    SC3 --> O1
    SG3 --> O2
    FE3 --> O3
    
    style Input fill:#e1f5ff
    style Classification fill:#FFB6C1
    style Segmentation fill:#87CEEB
    style FeatureExtraction fill:#DDA0DD
    style Scoring fill:#FFD700
    style Output fill:#90EE90
```

## Usage

These flowcharts can be:
1. **Rendered in Markdown viewers** (GitHub, GitLab, etc.)
2. **Exported to images** using Mermaid CLI or online tools
3. **Included in documentation** or presentations
4. **Referenced in the LaTeX report** (after conversion to images)

To convert to images:
```bash
# Install Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Convert to PNG
mmdc -i flowcharts.md -o flowcharts.png

# Convert individual diagrams
mmdc -i flowcharts.md -o flowcharts.pdf
```

