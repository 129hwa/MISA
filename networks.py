# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import wide_resnet
import copy

import timm


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        x = self.activation(x) # for URM; does not affect other algorithms
        return x

class DinoV2(torch.nn.Module):
    """ """
    def __init__(self,input_shape, hparams):
        super(DinoV2, self).__init__()

        self.network = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.n_outputs =  5 * 768

        nc = input_shape[0]

        if nc != 3:
            raise RuntimeError("Inputs must have 3 channels")

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['vit_dropout'])

        if hparams["vit_attn_tune"]:
            for n,p in self.network.named_parameters():
                if 'attn' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False


    def forward(self, x):
        x = self.network.get_intermediate_layers(x, n=4, return_class_token=True)
        linear_input = torch.cat([
            x[0][1],
            x[1][1],
            x[2][1],
            x[3][1],
            x[3][0].mean(1)
            ], dim=1)
        return self.dropout(linear_input)


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        if hparams['resnet50_augmix']:
            self.network = timm.create_model('resnet50.ram_in1k', pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        if hparams["freeze_bn"]:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.activation(self.dropout(self.network(x)))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.hparams["freeze_bn"]:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return self.activation(x)


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        if hparams["vit"]:
            if hparams["dinov2"]:
                return DinoV2(input_shape, hparams)
            else:
                raise NotImplementedError
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)

#######################################################
# Gradient Reversal Layer
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

# Domain Information Encoder
class DomainInformationEncoder(nn.Module):
    def __init__(self, feature_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), 
            nn.BatchNorm1d(feature_dim // 2), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate), 
            nn.Linear(feature_dim // 2, output_dim)
        )
    
    def forward(self, features):
        return self.encoder(features)

# Feature Attention Module
class FeatureAttention(nn.Module):
    def __init__(self, feature_dim, context_dim, num_heads, output_dim, dropout_rate=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads, 
            kdim=context_dim, 
            vdim=context_dim,
            batch_first=True, 
            dropout=dropout_rate
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, output_dim)
        )
        self.norm2 = nn.LayerNorm(output_dim)
    
    def forward(self, z_query_features, context_kv_features):
        z_q_unsqueeze = z_query_features.unsqueeze(1)
        context_kv_unsqueeze = context_kv_features.unsqueeze(1)
        attn_output, _ = self.attention(z_q_unsqueeze, context_kv_unsqueeze, context_kv_unsqueeze)
        attn_output = attn_output.squeeze(1)
        z_after_attn_norm = self.norm1(z_query_features + attn_output)
        mlp_output = self.mlp(z_after_attn_norm)
        return self.norm2(mlp_output)

# MIAS Model with CLIP Backbone
class MISA_DG_Model_CLIP(nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MISA_DG_Model_CLIP, self).__init__()
        
        # 모델 구성 처리 및 디버깅 정보
        self.debug_mode = True
        print(f"MISA_DG_Model_CLIP 초기화 - 입력 shape: {input_shape}, 클래스: {num_classes}, 도메인: {num_domains}")
        print(f"하이퍼파라미터: {hparams}")
        
        # CLIP 모델 로드
        clip_model_name = hparams['clip_model_name']
        try:
            self.clip_model, self.preprocess = clip.load(clip_model_name, device='cuda' if torch.cuda.is_available() else 'cpu', jit=False)
            print(f"CLIP 모델 '{clip_model_name}' 로드 성공")
        except Exception as e:
            raise RuntimeError(f"CLIP 모델 로드 오류: {e}")
        
        # 출력 차원 결정
        dummy_resolution = self.clip_model.visual.input_resolution
        dummy_image = torch.randn(1, 3, dummy_resolution, dummy_resolution)
        dummy_image = dummy_image.to(next(self.clip_model.parameters()).device)
        
        # CLIP 모델의 dtype 확인
        if hasattr(self.clip_model.visual, 'conv1') and hasattr(self.clip_model.visual.conv1, 'weight'):
            clip_dtype = self.clip_model.visual.conv1.weight.dtype
        elif hasattr(self.clip_model.visual, 'class_embedding'):
            clip_dtype = self.clip_model.visual.class_embedding.dtype
        else:
            clip_dtype = torch.float32
            
        with torch.no_grad():
            dummy_output = self.clip_model.encode_image(dummy_image.to(clip_dtype))
        
        clip_output_dim = dummy_output.shape[-1]
        self.latent_dim = hparams['latent_dim']
        print(f"CLIP 출력 차원: {clip_output_dim}, MIAS 잠재 차원: {self.latent_dim}")
        
        # 백본 고정 여부
        self.freeze_backbone = hparams['freeze_clip_backbone']
        if self.freeze_backbone:
            print("CLIP 백본 파라미터 고정")
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            print("CLIP 백본 파라미터 훈련 가능")
                
        # MIAS 컴포넌트
        context_vector_dim = clip_output_dim // 4
        self.dropout_rate = hparams['dropout_rate']
        
        # Domain Information Encoder
        self.die = DomainInformationEncoder(
            clip_output_dim, 
            context_vector_dim, 
            dropout_rate=self.dropout_rate
        )
        
        # Invariant Feature Attention
        self.ifa = FeatureAttention(
            clip_output_dim, 
            context_vector_dim, 
            hparams['attention_heads'], 
            self.latent_dim, 
            dropout_rate=self.dropout_rate
        )
        
        # Specific Feature Attention
        self.sfa = FeatureAttention(
            clip_output_dim, 
            context_vector_dim, 
            hparams['attention_heads'], 
            self.latent_dim, 
            dropout_rate=self.dropout_rate
        )
        
        # 분류기
        self.main_classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.BatchNorm1d(self.latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.latent_dim // 2, num_classes)
        )
        
        # 도메인 분류기
        self.domain_classifier_inv = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.BatchNorm1d(self.latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.latent_dim // 2, num_domains)
        )
        
        self.domain_classifier_spc = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.BatchNorm1d(self.latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.latent_dim // 2, num_domains)
        )
        
        # 재구성 네트워크
        self.reconstructor = nn.Sequential(
            nn.Linear(self.latent_dim * 2, clip_output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(clip_output_dim, clip_output_dim)
        )
        
        print(f"MIAS 모델 초기화 완료")
        
    def preprocess_input(self, x):
        """DomainBed 입력을 CLIP 형식으로 변환"""
        # 원본 이미지 크기 확인 및 로깅
        if self.debug_mode and not hasattr(self, 'input_shape_logged'):
            print(f"입력 이미지 원본 shape: {x.shape}")
            self.input_shape_logged = True
        
        # 이미지 크기 조정이 필요한 경우
        if x.shape[-1] != self.clip_model.visual.input_resolution or x.shape[-2] != self.clip_model.visual.input_resolution:
            x = F.interpolate(
                x, 
                size=(self.clip_model.visual.input_resolution, self.clip_model.visual.input_resolution), 
                mode='bilinear', 
                align_corners=False
            )
            
            if self.debug_mode and not hasattr(self, 'resized_shape_logged'):
                print(f"리사이즈된 이미지 shape: {x.shape}")
                self.resized_shape_logged = True
        
        # 이미지 정규화가 필요한 경우 추가 (DomainBed와 CLIP의 정규화 차이 처리)
        # 정규화 코드는 필요에 따라 여기에 추가...
        
        return x
        
    def forward(self, x, alpha=1.0):
        """
        MIAS 모델 포워드 패스
        Args:
            x: 입력 이미지 배치
            alpha: GRL 알파 값
        Returns:
            dictionary: 모델 출력 컴포넌트들
        """
        # 입력 전처리
        x = self.preprocess_input(x)
        
        # 적절한 데이터 유형 결정
        if hasattr(self.clip_model.visual, 'conv1') and hasattr(self.clip_model.visual.conv1, 'weight'):
            clip_dtype = self.clip_model.visual.conv1.weight.dtype
        elif hasattr(self.clip_model.visual, 'class_embedding'):
            clip_dtype = self.clip_model.visual.class_embedding.dtype
        else:
            clip_dtype = torch.float32
            
        # 백본이 고정되었거나 평가 모드인 경우
        if self.freeze_backbone or not self.training:
            with torch.no_grad():
                z_intermediate = self.clip_model.encode_image(x.to(clip_dtype))
        else:
            z_intermediate = self.clip_model.encode_image(x.to(clip_dtype))
        
        z_intermediate = z_intermediate.float()  # float32로 변환
        
        # 도메인 컨텍스트 추출
        domain_context = self.die(z_intermediate)
        
        # Invariant Feature Attention 적용
        z_inv = self.ifa(z_intermediate, domain_context)
        
        # Specific Feature Attention 적용
        z_spc = self.sfa(z_intermediate, domain_context)
        
        # 분류기 적용
        logits_task = self.main_classifier(z_inv)
        
        # 도메인 분류기 (GRL 적용)
        z_inv_reversed = grad_reverse(z_inv, alpha)
        logits_domain_inv = self.domain_classifier_inv(z_inv_reversed)
        
        # 도메인 분류기 (도메인 특이적 특징)
        logits_domain_spc = self.domain_classifier_spc(z_spc)
        
        # 재구성
        z_combined = torch.cat([z_inv, z_spc], dim=1)
        z_reconstructed = self.reconstructor(z_combined)
        
        # 디버깅 출력 (최초 실행 시)
        if self.debug_mode and not hasattr(self, 'forward_logged'):
            batch_size = x.size(0)
            print(f"MIAS forward - 배치 크기: {batch_size}")
            print(f"z_intermediate shape: {z_intermediate.shape}")
            print(f"z_inv shape: {z_inv.shape}")
            print(f"z_spc shape: {z_spc.shape}")
            print(f"logits_task shape: {logits_task.shape}")
            self.forward_logged = True
        
        return {
            'logits_task': logits_task,
            'logits_domain_inv': logits_domain_inv,
            'logits_domain_spc': logits_domain_spc,
            'z_inv': z_inv,
            'z_spc': z_spc,
            'z_intermediate': z_intermediate,
            'z_reconstructed': z_reconstructed
        }

class CLIPFeaturizer(torch.nn.Module):
    """CLIP-based feature extractor"""
    def __init__(self, input_shape, hparams):
        super(CLIPFeaturizer, self).__init__()
        self.hparams = hparams
        
        # CLIP 모델 로드
        try:
            self.clip_model, self.clip_preprocess = clip.load(
                hparams.get('clip_model_name', 'ViT-B/32'), 
                device='cpu',  # 초기에는 CPU에 로드
                jit=False
            )
            
            # Visual encoder만 사용
            self.visual_encoder = self.clip_model.visual
            
            # 출력 차원 계산
            dummy_input = torch.randn(1, 3, 224, 224)
            if hasattr(self.visual_encoder, 'conv1') and hasattr(self.visual_encoder.conv1, 'weight'):
                # ResNet-like
                self.clip_dtype = self.visual_encoder.conv1.weight.dtype
            elif hasattr(self.visual_encoder, 'class_embedding'):
                # ViT-like
                self.clip_dtype = self.visual_encoder.class_embedding.dtype
            else:
                self.clip_dtype = torch.float32
                
            with torch.no_grad():
                dummy_output = self.visual_encoder(dummy_input.type(self.clip_dtype))
            self.n_outputs = dummy_output.shape[-1]
            
            # Freeze/Unfreeze 설정
            if hparams.get('freeze_clip', True):
                for param in self.visual_encoder.parameters():
                    param.requires_grad = False
                print(f"CLIP visual encoder frozen. Output dim: {self.n_outputs}")
            else:
                print(f"CLIP visual encoder fine-tuning enabled. Output dim: {self.n_outputs}")
                
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            print("Falling back to ResNet50")
            # Fallback to ResNet
            self.visual_encoder = None
            self.use_clip = False
            resnet = torchvision.models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.n_outputs = 2048
        else:
            self.use_clip = True
            
        self.dropout = nn.Dropout(hparams.get('dropout', 0.0))

    def forward(self, x):
        if self.use_clip:
            # CLIP의 dtype에 맞춰 입력 변환
            if self.visual_encoder.training or not self.hparams.get('freeze_clip', True):
                features = self.visual_encoder(x.type(self.clip_dtype))
            else:
                with torch.no_grad():
                    features = self.visual_encoder(x.type(self.clip_dtype))
            features = features.float()  # 다음 layer를 위해 float32로 변환
        else:
            # Fallback ResNet
            features = self.backbone(x)
            features = torch.flatten(features, 1)
            
        return self.dropout(features)

    def train(self, mode=True):
        """Override train method to handle frozen CLIP"""
        super().train(mode)
        if self.use_clip and self.hparams.get('freeze_clip', True):
            self.visual_encoder.eval()  # Keep frozen CLIP in eval mode

# Featurizer 함수 수정
def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    # CLIP 사용 여부 체크
    if hparams.get('use_clip', False):
        return CLIPFeaturizer(input_shape, hparams)
    elif len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        if hparams["vit"]:
            if hparams["dinov2"]:
                return DinoV2(input_shape, hparams)
            else:
                raise NotImplementedError
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


# (선택) domainbed/networks.py 에 다음 클래스 추가

import clip

class CLIPFeaturizer(nn.Module):
    def __init__(self, input_shape, hparams):
        super(CLIPFeaturizer, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        # CLIP 모델의 vision output dimension
        self.n_outputs = self.clip_model.visual.output_dim

    def forward(self, x):
        # DomainBed의 입력 이미지(x)는 이미 [B,3,H,W] normalized 상태로 넘어옴을 가정
        # CLIP 이미지 인코더의 forward는 normalize + resize 논리 포함되어 있지 않으므로,
        # DomainBed 데이터셋 측에서 CLIP preprocessing을 동일하게 맞추어야 함.
        with torch.no_grad():
            # x가 이미 clip preprocessing을 거쳤다면 바로 encode_image 호출
            return self.clip_model.encode_image(x)
