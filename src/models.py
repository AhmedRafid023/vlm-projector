# import torch
# import torch.nn as nn
# import lightning as L
# from transformers import AutoModelForCausalLM

# class MedicalVLMExperiment(L.LightningModule):
#     def __init__(self, config):
#         super().__init__()
#         self.save_hyperparameters()
#         self.config = config
#         self.situation = config['experiment']['situation']

#         model_id = config['model']['id']
#         print(f"Loading {model_id} for Situation: {self.situation}...")

#         self.full_model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True,
#             device_map="auto"
#         )

#         # -------------------------------------------------
#         # 1) GLOBAL FREEZE
#         # -------------------------------------------------
#         for param in self.full_model.parameters():
#             param.requires_grad = False

#         # -------------------------------------------------
#         # 2) COMPONENT & DIMENSION DETECTION
#         # -------------------------------------------------
#         if hasattr(self.full_model, "model") and hasattr(self.full_model.model, "vision_tower"):
#             self.vision_tower = self.full_model.model.vision_tower
#         else:
#             self.vision_tower = getattr(self.full_model, "vision_tower", None)

#         self.projector = None
#         for name, module in self.full_model.named_modules():
#             if any(k in name.lower() for k in ["projector", "vision_proj", "mm_projector"]):
#                 self.projector = module
#                 break

#         # Dims
#         self.vision_dim = self.full_model.config.vision_config.hidden_size if hasattr(self.full_model.config, "vision_config") else 1152
#         self.text_hidden_dim = self.full_model.config.text_config.hidden_size if hasattr(self.full_model.config, "text_config") else 2560

#         # -------------------------------------------------
#         # 3) DYNAMIC ARCHITECTURE SETUP
#         # -------------------------------------------------
#         if self.situation == "vision_only":
#             # CHOICE A: Bypass Projector. Use raw 1152 features.
#             print("Mode: Vision Only (Choice A - Bypassing Projector)")
#             self.feature_dim = self.vision_dim
#             # No unfreezing needed (training only classifier)

#         elif self.situation == "vision_proj":
#             # CHOICE B: Use Projector. Use 2560 features.
#             print("Mode: Vision + Projector (Training Projector & Classifier)")
#             self.feature_dim = self.text_hidden_dim
#             if self.projector is not None:
#                 for param in self.projector.parameters():
#                     param.requires_grad = True
        
#         else:
#             raise ValueError(f"Unknown situation: {self.situation}")

#         # -------------------------------------------------
#         # 4) CLASSIFIER
#         # -------------------------------------------------
#         self.classifier = nn.Linear(self.feature_dim, config['model']['num_classes'])
#         self.loss_fn = nn.BCEWithLogitsLoss()

#         trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         print(f"Active Feature Dim: {self.feature_dim} | Trainable Params: {trainable:,}")

#     def forward(self, pixel_values):
#         # 1. Vision Features [B, Tokens, 1152]
#         out = self.vision_tower(pixel_values)
#         feat = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

#         # 2. Conditional Projector Path
#         if self.situation == "vision_proj" and self.projector is not None:
#             # Go through projector: 1152 -> 2560
#             feat = self.projector(feat)
#             if hasattr(feat, "last_hidden_state"):
#                 feat = feat.last_hidden_state
#             elif isinstance(feat, (list, tuple)):
#                 feat = feat[0]
        
#         # If situation is "vision_only", we skip the projector and stay at 1152

#         # 3. Mean Pool
#         feat = torch.mean(feat, dim=1)

#         # 4. Classify
#         return self.classifier(feat.float())

#     def training_step(self, batch, batch_idx):
#         logits = self(batch['pixel_values'])
#         loss = self.loss_fn(logits, batch['label'])
#         self.log("train_loss", loss, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         logits = self(batch['pixel_values'])
#         loss = self.loss_fn(logits, batch['label'])
#         self.log("val_loss", loss, prog_bar=True)

#     def configure_optimizers(self):
#         return torch.optim.AdamW(
#             filter(lambda p: p.requires_grad, self.parameters()),
#             lr=self.config['trainer']['learning_rate']
#         )



import torch
import torch.nn as nn
import lightning as L
from transformers import AutoModelForCausalLM, LlavaForConditionalGeneration

class MedicalVLMExperiment(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.situation = config['experiment']['situation']
        model_id = config['model']['id']

        # --- 1. UNIFIED LOADER ---
        # LLaVA models usually need LlavaForConditionalGeneration to load the vision tower
        if "llava" in model_id.lower():
            loader = LlavaForConditionalGeneration
        else:
            loader = AutoModelForCausalLM

        self.full_model = loader.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )

        # Global Freeze
        for param in self.full_model.parameters():
            param.requires_grad = False

        # --- 2. ROBUST COMPONENT DETECTION ---
        # MedGemma path
        if hasattr(self.full_model, "model") and hasattr(self.full_model.model, "vision_tower"):
            self.vision_tower = self.full_model.model.vision_tower
        # LLaVA-Med path
        elif hasattr(self.full_model, "vision_tower"):
            self.vision_tower = self.full_model.vision_tower
        else:
            self.vision_tower = getattr(self.full_model, "vision_tower", None)

        # Unified Projector Detection
        self.projector = None
        for name, module in self.full_model.named_modules():
            # Matches 'multi_modal_projector' (LLaVA) or 'vision_proj' (MedGemma)
            if any(k in name.lower() for k in ["projector", "vision_proj", "mm_projector"]):
                print(f"Detected Projector: {name}")
                self.projector = module
                break

        # --- 3. DYNAMIC DIMENSIONS ---
        # Vision Dim: SigLIP (1152) vs CLIP (1024)
        if hasattr(self.full_model.config, "vision_config"):
            self.vision_dim = self.full_model.config.vision_config.hidden_size
        else:
            self.vision_dim = 1152 if "gemma" in model_id.lower() else 1024

        # Text Dim: Gemma (2560) vs Mistral (4096)
        if hasattr(self.full_model.config, "text_config"):
            self.text_hidden_dim = self.full_model.config.text_config.hidden_size
        else:
            # Fallback for standard LLaVA configs
            self.text_hidden_dim = getattr(self.full_model.config, "hidden_size", 4096)

        # --- 4. SITUATION SETUP ---
        if self.situation == "vision_only":
            self.feature_dim = self.vision_dim
        elif self.situation == "vision_proj":
            self.feature_dim = self.text_hidden_dim
            if self.projector:
                for param in self.projector.parameters():
                    param.requires_grad = True
        
        self.classifier = nn.Linear(self.feature_dim, config['model']['num_classes'])
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pixel_values):
        out = self.vision_tower(pixel_values)
        feat = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

        if self.situation == "vision_proj" and self.projector:
            feat = self.projector(feat)
            # LLaVA projectors sometimes return objects
            if hasattr(feat, "last_hidden_state"): feat = feat.last_hidden_state
            elif isinstance(feat, (list, tuple)): feat = feat[0]

        feat = torch.mean(feat, dim=1)
        return self.classifier(feat.float())

    def training_step(self, batch, batch_idx):
        logits = self(batch['pixel_values'])
        loss = self.loss_fn(logits, batch['label'])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['pixel_values'])
        loss = self.loss_fn(logits, batch['label'])
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                 lr=self.config['trainer']['learning_rate'])