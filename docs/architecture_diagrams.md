# Architecture Diagrams

## 1. Flagship Screening Pipeline

```mermaid
graph LR
    Input[Input MRI slice] --> Preprocess[Resize to 224x224 grayscale tensor]
    Preprocess --> Detect[Residual CNN detector]
    Detect --> Threshold{Probability >= threshold}
    Threshold -->|No| Normal[Return normal / low-risk screen]
    Threshold -->|Yes| Classify[EfficientNet-based subtype classifier]
    Classify --> Output[Subtype distribution + top class]
```

## 2. Core Detection Model

```mermaid
graph TD
    Input[224x224x1 MRI] --> Stem[Conv + BatchNorm + MaxPool]
    Stem --> RB1[Residual block + channel attention]
    RB1 --> RB2[Residual block + channel attention]
    RB2 --> RB3[Residual block + channel attention]
    RB3 --> Head[Global average pooling + dense head]
    Head --> Out[Sigmoid tumour probability]
```

## 3. Core Classification Model

```mermaid
graph TD
    Input[224x224x1 MRI] --> Adapt[1x1 conv to 3 channels]
    Adapt --> Backbone[EfficientNetB0 backbone]
    Backbone --> Attention[Squeeze-and-excitation attention]
    Attention --> DenseHead[Dense + BatchNorm + Dropout]
    DenseHead --> Out[4-class softmax]
```

## 4. Experimental Segmentation Track

```mermaid
graph TD
    Input[Input MRI 128x128x1] --> C1[Residual encoder block]
    C1 --> P1[MaxPool]
    P1 --> C2[Residual encoder block]
    C2 --> P2[MaxPool]
    P2 --> C3[Residual encoder block]
    C3 --> P3[MaxPool]
    P3 --> B[Bottleneck block]
    B --> U1[Upsample + attention gate]
    U1 --> U2[Upsample + attention gate]
    U2 --> U3[Upsample + attention gate]
    U3 --> Out[Sigmoid mask]
```

## 5. Experimental Conditional GAN

```mermaid
graph LR
    subgraph Generator
        Noise[Latent vector z] --> MergeG[Concatenate with class condition]
        Label[One-hot class] --> MergeG
        MergeG --> DenseG[Dense + reshape]
        DenseG --> Up1[Transpose conv / residual upsampling]
        Up1 --> Up2[Feature refinement]
        Up2 --> Fake[Generated MRI]
    end

    subgraph Discriminator
        Fake --> DiscIn[Conditional discriminator]
        Real[Real MRI] --> DiscIn
        LabelD[One-hot class] --> DiscIn
        DiscIn --> Score[Real / fake score]
    end
```

## 6. Deployment Topology

```mermaid
graph LR
    Dev[Developer workstation] --> Git[Git + Git LFS repo]
    Git --> Docker[Docker image build]
    Git --> Ansible[Ansible automation]
    Docker --> Compose[Docker Compose runtime]
    Ansible --> Compose
    Docker --> K8s[Kubernetes deployment]
    K8s --> App[Streamlit MRI service]
    Compose --> App
```
