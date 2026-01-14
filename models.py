#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FedMKD Model Definition Module

Provides multi-modal cross-attention models and knowledge distillation loss functions,
supporting multiple datasets such as GAMMA, Zhongshan, Gongli,
implementing cross-attention-based feature fusion and knowledge distillation in federated learning scenarios.

Main functions:
    - LightweightOCTBranch: Lightweight OCT image branch
    - CrossAttentionFusion: Cross-attention fusion module
    - MultiModalCrossAttentionFusion: Multi-modal cross-attention fusion module
    - Multiple predefined multi-modal model architectures
    - KnowledgeDistillationLoss: Knowledge distillation loss function
    - get_model: Model factory function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Dict, Any


class LightweightOCTBranch(nn.Module):
    """
    Lightweight OCT Image Branch Model
    
    Used for processing OCT 3D image sequences, including:
    - Slice-level feature extraction
    - Sequence aggregation processing
    
    Args:
        num_slices: Number of slices in input OCT image, default 256
        output_features: Output feature dimension, default 512
        
    Input shape:
        [batch_size, num_slices, 3, height, width]
        
    Output shape:
        [batch_size, output_features]
    """
    
    def __init__(self, num_slices: int = 256, output_features: int = 512):
        super(LightweightOCTBranch, self).__init__()
        
        self.slice_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.sequence_aggregation = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.output_dim = 512
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Propagation
        
        Args:
            x: Input tensor, shape [batch_size, num_slices, 3, H, W]
            
        Returns:
            Output features, shape [batch_size, 512]
        """
        batch_size, num_slices, channels, height, width = x.shape
        
        step = max(1, num_slices // 16)
        selected_indices = torch.arange(0, num_slices, step, device=x.device)[:16]
        x_selected = x[:, selected_indices]
        
        selected_slices = x_selected.shape[1]
        x_reshaped = x_selected.view(batch_size * selected_slices, channels, height, width)
        
        slice_features = self.slice_conv(x_reshaped)
        slice_features = slice_features.view(batch_size * selected_slices, -1)
        slice_features = slice_features.view(batch_size, selected_slices, -1)
        
        aggregated_features = torch.mean(slice_features, dim=1)
        output_features = self.sequence_aggregation(aggregated_features)
        
        return output_features


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Module
    
    Supports two gating mechanisms:
    - sigmoid: Complementary gating, sum of two modal weights is 1
    - softmax: Probability gating, naturally ensures sum of weights is 1
    
    Args:
        fundus_dim: Fundus feature dimension, default 512
        oct_dim: OCT feature dimension, default 512
        hidden_dim: Hidden layer dimension, default 256
        num_heads: Number of attention heads, default 8
        use_gating: Whether to use gating mechanism, default True
        gate_type: Gating type, optional 'sigmoid' or 'softmax', default 'sigmoid'
    """
    
    def __init__(
        self,
        fundus_dim: int = 512,
        oct_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8,
        use_gating: bool = True,
        gate_type: str = 'sigmoid'
    ):
        super(CrossAttentionFusion, self).__init__()
        
        self.fundus_dim = fundus_dim
        self.oct_dim = oct_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_gating = use_gating
        self.gate_type = gate_type
        
        self.fundus_proj = nn.Linear(fundus_dim, hidden_dim)
        self.oct_proj = nn.Linear(oct_dim, hidden_dim)
        
        self.cross_attention_f2o = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_attention_o2f = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        
        self.ln_fundus = nn.LayerNorm(hidden_dim)
        self.ln_oct = nn.LayerNorm(hidden_dim)
        
        self.ffn_fundus = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.ffn_oct = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        if self.use_gating:
            output_dim = 1 if self.gate_type == 'sigmoid' else 2
            self.gate_network = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            )
        
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(
        self,
        fundus_features: torch.Tensor,
        oct_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward Propagation
        
        Args:
            fundus_features: Fundus features, shape [batch_size, fundus_dim]
            oct_features: OCT features, shape [batch_size, oct_dim]
            
        Returns:
            Fused features, shape [batch_size, hidden_dim]
        """
        fundus_proj = self.fundus_proj(fundus_features).unsqueeze(1)
        oct_proj = self.oct_proj(oct_features).unsqueeze(1)
        
        fundus_attended, _ = self.cross_attention_f2o(
            query=fundus_proj, key=oct_proj, value=oct_proj
        )
        fundus_attended = self.ln_fundus(fundus_attended + fundus_proj)
        fundus_attended = fundus_attended + self.ffn_fundus(fundus_attended)
        
        oct_attended, _ = self.cross_attention_o2f(
            query=oct_proj, key=fundus_proj, value=fundus_proj
        )
        oct_attended = self.ln_oct(oct_attended + oct_proj)
        oct_attended = oct_attended + self.ffn_oct(oct_attended)
        
        fundus_final = fundus_attended.squeeze(1)
        oct_final = oct_attended.squeeze(1)
        
        if self.use_gating:
            concat_features = torch.cat([fundus_final, oct_final], dim=1)
            gate_output = self.gate_network(concat_features)
            
            if self.gate_type == 'sigmoid':
                fundus_gate = torch.sigmoid(gate_output)
                oct_gate = 1.0 - fundus_gate
            else:
                gates = torch.softmax(gate_output, dim=1)
                fundus_gate = gates[:, 0:1]
                oct_gate = gates[:, 1:2]
            
            gated_fundus = fundus_gate * fundus_final
            gated_oct = oct_gate * oct_final
            fused_features = torch.cat([gated_fundus, gated_oct], dim=1)
        else:
            fused_features = torch.cat([fundus_final, oct_final], dim=1)
        
        output = self.output_proj(fused_features)
        return output


class MultiModalCrossAttentionFusion(nn.Module):
    """
    Multi-Modal Cross-Attention Fusion Module (Supports 4 Modals)
    
    Each modal serves as query, other modals as key and value,
    calculate correlation and fuse features.
    
    Args:
        fundus_dim: Fundus feature dimension, default 512
        oct_dev_dim: OCT_dev feature dimension, default 512
        oct_pie_dim: OCT_pie feature dimension, default 512
        vf_dim: VF feature dimension, default 512
        hidden_dim: Hidden layer dimension, default 256
        num_heads: Number of attention heads, default 8
    """
    
    def __init__(
        self,
        fundus_dim: int = 512,
        oct_dev_dim: int = 512,
        oct_pie_dim: int = 512,
        vf_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8
    ):
        super(MultiModalCrossAttentionFusion, self).__init__()
        
        self.fundus_dim = fundus_dim
        self.oct_dev_dim = oct_dev_dim
        self.oct_pie_dim = oct_pie_dim
        self.vf_dim = vf_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.fundus_proj = nn.Linear(fundus_dim, hidden_dim)
        self.oct_dev_proj = nn.Linear(oct_dev_dim, hidden_dim)
        self.oct_pie_proj = nn.Linear(oct_pie_dim, hidden_dim)
        self.vf_proj = nn.Linear(vf_dim, hidden_dim)
        
        self.cross_attention_fundus = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_attention_oct_dev = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_attention_oct_pie = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_attention_vf = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        
        self.ln_fundus = nn.LayerNorm(hidden_dim)
        self.ln_oct_dev = nn.LayerNorm(hidden_dim)
        self.ln_oct_pie = nn.LayerNorm(hidden_dim)
        self.ln_vf = nn.LayerNorm(hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        
    def forward(
        self,
        fundus_features: torch.Tensor,
        oct_dev_features: torch.Tensor,
        oct_pie_features: torch.Tensor,
        vf_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward Propagation
        
        Args:
            fundus_features: Fundus features, shape [batch_size, fundus_dim]
            oct_dev_features: OCT_dev features, shape [batch_size, oct_dev_dim]
            oct_pie_features: OCT_pie features, shape [batch_size, oct_pie_dim]
            vf_features: VF features, shape [batch_size, vf_dim]
            
        Returns:
            Fused features, shape [batch_size, hidden_dim]
        """
        fundus_proj = self.fundus_proj(fundus_features).unsqueeze(1)
        oct_dev_proj = self.oct_dev_proj(oct_dev_features).unsqueeze(1)
        oct_pie_proj = self.oct_pie_proj(oct_pie_features).unsqueeze(1)
        vf_proj = self.vf_proj(vf_features).unsqueeze(1)
        
        fundus_attended, _ = self.cross_attention_fundus(
            query=fundus_proj,
            key=torch.cat([oct_dev_proj, oct_pie_proj, vf_proj], dim=1),
            value=torch.cat([oct_dev_proj, oct_pie_proj, vf_proj], dim=1)
        )
        fundus_attended = self.ln_fundus(fundus_attended + fundus_proj)
        
        oct_dev_attended, _ = self.cross_attention_oct_dev(
            query=oct_dev_proj,
            key=torch.cat([fundus_proj, oct_pie_proj, vf_proj], dim=1),
            value=torch.cat([fundus_proj, oct_pie_proj, vf_proj], dim=1)
        )
        oct_dev_attended = self.ln_oct_dev(oct_dev_attended + oct_dev_proj)
        
        oct_pie_attended, _ = self.cross_attention_oct_pie(
            query=oct_pie_proj,
            key=torch.cat([fundus_proj, oct_dev_proj, vf_proj], dim=1),
            value=torch.cat([fundus_proj, oct_dev_proj, vf_proj], dim=1)
        )
        oct_pie_attended = self.ln_oct_pie(oct_pie_attended + oct_pie_proj)
        
        vf_attended, _ = self.cross_attention_vf(
            query=vf_proj,
            key=torch.cat([fundus_proj, oct_dev_proj, oct_pie_proj], dim=1),
            value=torch.cat([fundus_proj, oct_dev_proj, oct_pie_proj], dim=1)
        )
        vf_attended = self.ln_vf(vf_attended + vf_proj)
        
        final_fundus = fundus_attended.squeeze(1)
        final_oct_dev = oct_dev_attended.squeeze(1)
        final_oct_pie = oct_pie_attended.squeeze(1)
        final_vf = vf_attended.squeeze(1)
        
        fused_features = torch.cat(
            [final_fundus, final_oct_dev, final_oct_pie, final_vf], dim=1
        )
        output = self.output_proj(fused_features)
        
        return output


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss Function
    
    Combines hard label loss and soft label loss,
    implements knowledge transfer from teacher model to student model.
    
    Args:
        temperature: Temperature parameter for softening probability distribution, default 4.0
        alpha: Balance coefficient, weight between hard loss and soft loss, default 0.5
    
    Inputs:
        student_logits: Student model logits, shape [batch_size, num_classes]
        teacher_logits: Teacher model logits, shape [batch_size, num_classes]
        hard_labels: Hard labels, shape [batch_size]
        
    Outputs:
        total_loss: Total loss
        hard_loss: Hard loss (CrossEntropy)
        soft_loss: Soft loss (KLDivLoss)
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_labels: torch.Tensor
    ) -> tuple:
        """
        Calculate Knowledge Distillation Loss
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            hard_labels: Hard labels
            
        Returns:
            (total_loss, hard_loss, soft_loss) tuple
        """
        hard_loss = F.cross_entropy(student_logits, hard_labels)
        
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss, hard_loss, soft_loss


def _create_backbone(backbone_type: str = 'resnet'):
    """
    Create Backbone Network Branch

    Args:
        backbone_type: Backbone network type, 'resnet' or 'mobilenet'

    Returns:
        Backbone network branch module

    Raises:
        ValueError: Unsupported backbone network type
    """

    if backbone_type == 'resnet':
        resnet = models.resnet18(pretrained=True)
        return nn.Sequential(*list(resnet.children())[:-1])
    elif backbone_type == 'mobilenet':
        mobilenet = models.mobilenet_v2(pretrained=True)
        return mobilenet.features
    else:
        raise ValueError(f"Unsupported backbone network: {backbone_type}, only 'resnet' or 'mobilenet' are supported")


def _get_backbone_dim(backbone_type: str = 'resnet') -> int:
    """
    Get Backbone Network Output Feature Dimension

    Args:
        backbone_type: Backbone network type, 'resnet' or 'mobilenet'

    Returns:
        Output feature dimension
    """

    if backbone_type == 'resnet':
        return 512
    elif backbone_type == 'mobilenet':
        return 1280
    else:
        raise ValueError(f"Unsupported backbone network: {backbone_type}, only 'resnet' or 'mobilenet' are supported")


class Fundus_CrossAttention(nn.Module):
    """
    Fundus Image Classification Model

    Based on cross-attention mechanism, supports ResNet18 and MobileNetV2 backbone networks

    Args:
        num_classes: Number of classes, default 2
        backbone: Backbone network type, 'resnet' or 'mobilenet', default 'resnet'
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = 'resnet'
    ):
        super(Fundus_CrossAttention, self).__init__()

        self.backbone = backbone
        self.features = _create_backbone(backbone)
        backbone_dim = _get_backbone_dim(backbone)
        self.classifier = nn.Linear(backbone_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Propagation

        Args:
            x: Input image, shape [batch_size, 3, H, W]

        Returns:
            Classification logits, shape [batch_size, num_classes]
        """
        features = self.features(x)
        if self.backbone == 'mobilenet':
            features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)

        return logits


class Gongli_Dual_CrossAttention(nn.Module):
    """
    Gongli Dataset Dual-Modal Fusion Model

    Based on cross-attention mechanism, supports ResNet18 and MobileNetV2 backbone networks

    Args:
        num_classes: Number of classes, default 2
        hidden_dim: Hidden layer dimension, default 256
        num_heads: Number of attention heads, default 8
        backbone: Backbone network type, 'resnet' or 'mobilenet', default 'resnet'
    """

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 256,
        num_heads: int = 8,
        backbone: str = 'resnet'
    ):
        super(Gongli_Dual_CrossAttention, self).__init__()

        self.fundus_branch = _create_backbone(backbone)
        self.oct_gl_branch = _create_backbone(backbone)

        backbone_dim = _get_backbone_dim(backbone)

        self.cross_attention = CrossAttentionFusion(
            fundus_dim=backbone_dim, oct_dim=backbone_dim,
            hidden_dim=hidden_dim, num_heads=num_heads
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, inputs: tuple) -> torch.Tensor:
        """
        Forward pass

        Args:
            inputs: (fundus_img, oct_gl_img) tuple

        Returns:
            Classification logits, shape [batch_size, num_classes]
        """
        if isinstance(inputs, (list, tuple)):
            fundus_img, oct_gl_img = inputs
        else:
            raise ValueError("Input should be a tuple of (fundus_img, oct_gl_img)")

        fundus_features = self.fundus_branch(fundus_img)
        fundus_features = F.adaptive_avg_pool2d(fundus_features, 1)
        fundus_features = fundus_features.view(fundus_features.size(0), -1)

        oct_gl_features = self.oct_gl_branch(oct_gl_img)
        oct_gl_features = F.adaptive_avg_pool2d(oct_gl_features, 1)
        oct_gl_features = oct_gl_features.view(oct_gl_features.size(0), -1)

        fused_features = self.cross_attention(fundus_features, oct_gl_features)
        logits = self.classifier(fused_features)

        return logits


class Zhongshan_Quad_CrossAttention(nn.Module):
    """
    Zhongshan dataset quad-modal fusion model

    Based on cross-attention mechanism, supports ResNet18 and MobileNetV2 backbones,
    implements feature fusion for fundus images, OCT_dev, OCT_pie, and VF images.

    Args:
        num_classes: Number of classes, default 2
        hidden_dim: Hidden layer dimension, default 256
        num_heads: Number of attention heads, default 8
        backbone: Backbone network type, 'resnet' or 'mobilenet', default 'resnet'
    """

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 256,
        num_heads: int = 8,
        backbone: str = 'resnet'
    ):
        super(Zhongshan_Quad_CrossAttention, self).__init__()

        self.fundus_branch = _create_backbone(backbone)
        self.oct_dev_branch = _create_backbone(backbone)
        self.oct_pie_branch = _create_backbone(backbone)
        self.vf_branch = _create_backbone(backbone)

        backbone_dim = _get_backbone_dim(backbone)

        self.cross_attention = MultiModalCrossAttentionFusion(
            fundus_dim=backbone_dim,
            oct_dev_dim=backbone_dim,
            oct_pie_dim=backbone_dim,
            vf_dim=backbone_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, inputs: tuple) -> torch.Tensor:
        """
        Forward pass

        Args:
            inputs: (fundus_img, oct_dev_img, oct_pie_img, vf_img) tuple

        Returns:
            Classification logits, shape [batch_size, num_classes]
        """
        if isinstance(inputs, (list, tuple)):
            fundus_img, oct_dev_img, oct_pie_img, vf_img = inputs
        else:
            raise ValueError("Input should be a tuple of (fundus_img, oct_dev_img, oct_pie_img, vf_img)")

        fundus_features = self.fundus_branch(fundus_img)
        fundus_features = F.adaptive_avg_pool2d(fundus_features, 1)
        fundus_features = fundus_features.view(fundus_features.size(0), -1)

        oct_dev_features = self.oct_dev_branch(oct_dev_img)
        oct_dev_features = F.adaptive_avg_pool2d(oct_dev_features, 1)
        oct_dev_features = oct_dev_features.view(oct_dev_features.size(0), -1)

        oct_pie_features = self.oct_pie_branch(oct_pie_img)
        oct_pie_features = F.adaptive_avg_pool2d(oct_pie_features, 1)
        oct_pie_features = oct_pie_features.view(oct_pie_features.size(0), -1)

        vf_features = self.vf_branch(vf_img)
        vf_features = F.adaptive_avg_pool2d(vf_features, 1)
        vf_features = vf_features.view(vf_features.size(0), -1)

        fused_features = self.cross_attention(
            fundus_features, oct_dev_features, oct_pie_features, vf_features
        )
        logits = self.classifier(fused_features)

        return logits


class GAMMA_Dual_CrossAttention(nn.Module):
    """
    GAMMA dataset dual-modal fusion model

    Based on cross-attention mechanism, supports ResNet18 and MobileNetV2 backbones,
    implements feature fusion for fundus images and OCT images.

    Args:
        num_classes: Number of classes, default 2
        hidden_dim: Hidden layer dimension, default 256
        num_heads: Number of attention heads, default 8
        backbone: Backbone network type, 'resnet' or 'mobilenet', default 'resnet'
    """

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 256,
        num_heads: int = 8,
        backbone: str = 'resnet'
    ):
        super(GAMMA_Dual_CrossAttention, self).__init__()

        self.fundus_branch = _create_backbone(backbone)
        self.oct_branch = LightweightOCTBranch(num_slices=64, output_features=512)

        backbone_dim = _get_backbone_dim(backbone)

        self.cross_attention = CrossAttentionFusion(
            fundus_dim=backbone_dim, oct_dim=512,
            hidden_dim=hidden_dim, num_heads=num_heads
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, inputs: tuple) -> torch.Tensor:
        """
        Forward pass

        Args:
            inputs: (fundus_img, oct_img) tuple

        Returns:
            Classification logits, shape [batch_size, num_classes]
        """
        if isinstance(inputs, (list, tuple)):
            fundus_img, oct_img = inputs
        else:
            raise ValueError("Input should be a tuple of (fundus_img, oct_img)")

        fundus_features = self.fundus_branch(fundus_img)
        fundus_features = F.adaptive_avg_pool2d(fundus_features, 1)
        fundus_features = fundus_features.view(fundus_features.size(0), -1)

        oct_features = self.oct_branch(oct_img)
        fused_features = self.cross_attention(fundus_features, oct_features)
        logits = self.classifier(fused_features)

        return logits



def get_model(
    model_type: str = 'mm',
    backbone: str = 'resnet',
    dataset: str = 'gamma',
    num_classes: int = 2,
    use_unified: bool = True,
    **kwargs: Any
) -> nn.Module:
    """
    Model factory function

    Returns the corresponding model instance based on model type, backbone network, and dataset.

    Args:
        model_type: Model type
            - 'mm': Multi-modal model
            - 'um': Single-modal model
        backbone: Backbone network
            - 'resnet': ResNet18
            - 'mobilenet': MobileNetV2
        dataset: Dataset type, optional 'gamma', 'zhongshan', 'gongli', 'airogs', etc.
        num_classes: Number of classes, default 2
        use_unified: Whether to use unified model class (supports backbone parameter), default True
        **kwargs: Other keyword arguments

    Returns:
        Model instance

    Raises:
        ValueError: Unsupported model type, backbone network, or dataset type
    """
    if use_unified:
        unified_models = {
            'gamma': GAMMA_Dual_CrossAttention,
            'zhongshan': Zhongshan_Quad_CrossAttention,
            'gongli': Gongli_Dual_CrossAttention
        }
        mm_model = unified_models.get(dataset, GAMMA_Dual_CrossAttention)
        if model_type == 'mm':
            return mm_model(num_classes=num_classes, backbone=backbone, **kwargs)
        elif model_type == 'um':
            return Fundus_CrossAttention(backbone=backbone, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}, only 'mm' or 'um' are supported")
    else:
        raise ValueError("Non-unified mode is deprecated, please use use_unified=True")

