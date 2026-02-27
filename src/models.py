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
                    
        elif self.situation == "full_finetune":
            self.feature_dim = self.vision_dim
            if self.vision_tower:
                for param in self.vision_tower.parameters():
                    param.requires_grad = True
                    
        elif self.situation == "vision_proj_full":
            self.feature_dim = self.text_hidden_dim
            if self.vision_tower:
                for param in self.vision_tower.parameters():
                    param.requires_grad = True
            if self.projector:
                for param in self.projector.parameters():
                    param.requires_grad = True
        else:
            raise ValueError(f"Unknown situation: {self.situation}. Must be one of: "
                           "'vision_only', 'vision_proj', 'full_finetune', 'vision_proj_full'")
        
        self.classifier = nn.Linear(self.feature_dim, config['model']['num_classes'])
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Print trainable parameters summary
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Feature Dim: {self.feature_dim} | Trainable Params: {trainable:,} / {total:,}")

    def forward(self, pixel_values):
        out = self.vision_tower(pixel_values)
        feat = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

        # Conditionally apply projector for situations that use it
        if self.situation in ["vision_proj", "vision_proj_full"] and self.projector:
            feat = self.projector(feat)
            # LLaVA projectors sometimes return objects
            if hasattr(feat, "last_hidden_state"): 
                feat = feat.last_hidden_state
            elif isinstance(feat, (list, tuple)): 
                feat = feat[0]
        # For "full_finetune" and "vision_only", we use raw vision features

        # Mean pooling across tokens
        feat = torch.mean(feat, dim=1)
        
        # Classification
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
