import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, tanh: bool = True):
        super(FeatureExtractor, self).__init__()
        self.tanh = tanh
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(4096, output_dim, bias=False)
        )
        self._init_weights(input_dim, output_dim)

    def _init_weights(self, input_dim: int, output_dim: int):
        nn.init.uniform_(
            self.layers[-1].weight, 
            -1. / np.sqrt(np.float32(input_dim)), 
            1. / np.sqrt(np.float32(input_dim))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.layers(x)
        if self.tanh:
            features = torch.tanh(features)
        
        norm = torch.norm(features, p=2, dim=1, keepdim=True)
        return features / norm

class Embedding(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int):
        super(Embedding, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(num_classes, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, embedding_dim)
        )
        
        self._init_weights(embedding_dim, num_classes)

    def _init_weights(self, embedding_dim: int, num_classes: int):
        nn.init.uniform_(
            self.layers[-1].weight, 
            -1. / np.sqrt(np.float32(num_classes)), 
            1. / np.sqrt(np.float32(num_classes))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.layers(x)
        embeddings = torch.tanh(embeddings)
        
        norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        return embeddings / norm

class ModalConsensusModule(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 2048):
        super().__init__()
        self.feat_dim = feat_dim
        self.attention = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        
    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor):
        assert img_feat.shape[1] == self.feat_dim, f"Image feature dimension {img_feat.shape[1]} does not match expected {self.feat_dim}"
        assert text_feat.shape[1] == self.feat_dim, f"Text feature dimension {text_feat.shape[1]} does not match expected {self.feat_dim}"
        
        feat_diff = torch.abs(img_feat - text_feat)
        concat_feat = torch.cat([img_feat, text_feat], dim=1)
        consensus_weight = self.attention(concat_feat)
        
        consensus_weight = torch.where(feat_diff > 0.5, consensus_weight * 0.8, consensus_weight)
        
        img_feat_weighted = img_feat * consensus_weight
        text_feat_weighted = text_feat * consensus_weight
        
        consensus_feat = self.fusion(torch.cat([img_feat_weighted, text_feat_weighted], dim=1))
        consensus_feat = F.normalize(consensus_feat, p=2, dim=1)
        
        return img_feat_weighted, text_feat_weighted, consensus_feat, consensus_weight

class AdaptiveTempClassifier(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes, bias=False)
        self.temp = nn.Parameter(torch.ones(1) * 1.0)
    
    def forward(self, feat: torch.Tensor):
        logits = self.fc(feat)
        temp = F.softplus(self.temp) + 0.1
        pred = F.softmax(logits / temp, dim=1)
        return pred, logits, temp

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()

        vgg = models.vgg19_bn(pretrained=True)
        self.features = vgg.features
        self.classifier_features = nn.Sequential(*list(vgg.classifier.children())[:-2])
        
        for i, param in enumerate(self.features.parameters()):
            if i < 10:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_features = self.features(x)
        flattened = conv_features.view(x.size(0), -1)
        final_features = self.classifier_features(flattened)
        return final_features

class CMNN(nn.Module):
    def __init__(self, img_input_dim: int = 4096, text_input_dim: int = 1024, 
                 output_dim: int = 1024, num_class: int = 10, tanh: bool = True):
        super(CMNN, self).__init__()
        
        self.img_net = FeatureExtractor(img_input_dim, output_dim, tanh)
        self.text_net = FeatureExtractor(text_input_dim, output_dim, tanh)
        
        self.consensus_module = ModalConsensusModule(output_dim)
        
        self.img_classifier = AdaptiveTempClassifier(output_dim, num_class)
        self.text_classifier = AdaptiveTempClassifier(output_dim, num_class)
        
        self.gate = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor, text: torch.Tensor, return_consensus: bool = True):
        img_feat = self.img_net(img)
        text_feat = self.text_net(text)
        
        img_feat_weighted, text_feat_weighted, consensus_feat, consensus_weight = self.consensus_module(img_feat, text_feat)
        
        img_gate = self.gate(img_feat_weighted)
        text_gate = self.gate(text_feat_weighted)
        img_feat_gated = img_feat_weighted * img_gate
        text_feat_gated = text_feat_weighted * text_gate
        
        img_pred, img_logits, img_temp = self.img_classifier(img_feat_gated)
        text_pred, text_logits, text_temp = self.text_classifier(text_feat_gated)
        
        img_feat_final = F.normalize(img_feat_gated, p=2, dim=1)
        text_feat_final = F.normalize(text_feat_gated, p=2, dim=1)
        
        if return_consensus:
            return (img_feat_final, text_feat_final), (img_pred, text_pred), (consensus_feat, consensus_weight)
        else:
            return img_feat_final, text_feat_final

class CMNN_Compat(CMNN):
    def forward(self, img: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_feat, text_feat = super().forward(img, text, return_consensus=False)
        return img_feat, text_feat

class CMNN_Base(nn.Module):
    def __init__(self, img_input_dim: int = 4096, text_input_dim: int = 1024, 
                 output_dim: int = 1024, num_class: int = 10, tanh: bool = True):
        super(CMNN_Base, self).__init__()
        
        self.img_net = FeatureExtractor(img_input_dim, output_dim, tanh)
        self.text_net = FeatureExtractor(text_input_dim, output_dim, tanh)

    def forward(self, img: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_feature = self.img_net(img)
        text_feature = self.text_net(text)
        return img_feature, text_feature